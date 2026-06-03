import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import *
from torch_geometric.nn import LayerNorm
import pandas as pd
import math


import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        # x: (B, F, C)
        w = x.mean(dim=1)             # (B, C)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))# (B, C)
        w = w.unsqueeze(1)            # (B, 1, C)
        return x * w


class AttentionWrapper(nn.Module):
    def __init__(self,group_dim):
        super().__init__()
        self.num_groups = 104
        self.group_dim = group_dim
        
        # 原来的 Channel Attention
        self.channel_attention = ChannelAttention(
            channels=self.group_dim,
            reduction=1
        )

        # ⭐ 新增：3 个可学习的 q，每个 shape 为 (3,)
        # q_list 是 (3, 3) 的参数矩阵，每行一个 q
        self.q = nn.Parameter(torch.randn(self.group_dim, self.group_dim))

    def forward(self, x):
        # x: (B, 312) → reshape 为 (B,104,3)
        B = x.size(0)
        x = x.view(B, self.num_groups, self.group_dim)   # (B,104,3)

        # 1. 先做 channel attention
        x = self.channel_attention(x)                    # (B,104,3)

        # 2. 计算权重
        #    q: (3,3) → 3 个 query seed，每个 3 维
        weights = F.softmax(self.q, dim=-1)              # (3,3)

        # 3. 用 3 个 q 分别聚合 → 得到 3 个通道
        #    x: (B,104,3)
        #    weights: (3,3) → (1,1,3,3)

        # 扩展维度便于广播
        w = weights.unsqueeze(1).unsqueeze(1)            # (3,1,1,3)

        # 输出的每个通道是：
        # out[k] = sum_j x[...,j] * w[k,j]
        # 得到 shape: (B,104,3)
        out = torch.einsum('bfc,kc->bfk', x, weights)
        

        return out   # (B,104,3)#TODO 
#TODO 原来的可学习权重



class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        assert query_dim % num_heads == 0, "query_dim 必须能被 num_heads 整除"
        
        self.num_heads = num_heads
        self.dk = query_dim // num_heads  # 每个头的维度

        # 不同线性层映射 Query、Key、Value
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(key_dim, query_dim)
        self.value_proj = nn.Linear(key_dim, query_dim)

        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query: (B, Lq, Fq)  → 细胞系特征 (B, 3, 104)
        key/value: (B, Lk, Fk) → 药物特征 (B, 3, 512)
        """
        B, Lq, _ = query.shape
        _, Lk, _ = key.shape

        # 线性变换
        Q = self.query_proj(query)      # (B, Lq, Fq)
        K = self.key_proj(key)          # (B, Lk, Fq)
        V = self.value_proj(value)      # (B, Lk, Fq)

        # 分多头
        Q = Q.view(B, Lq, self.num_heads, self.dk).transpose(1, 2)  # (B, heads, Lq, dk)
        K = K.view(B, Lk, self.num_heads, self.dk).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.dk).transpose(1, 2)

        # 注意力权重计算
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.dk ** 0.5)  # (B, heads, Lq, Lk)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和更新细胞系表示
        context = torch.matmul(attn_weights, V)  # (B, heads, Lq, dk)
        context = context.transpose(1, 2).contiguous().view(B, Lq, -1)  # (B, Lq, Fq)

        # 输出融合
        output = self.out_proj(context)  # (B, Lq, Fq)
        return output, attn_weights

class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=4, dropout_prob=0.1):
        super().__init__()
        self.self_attention = SelfAttention(hidden_size, num_attention_heads, dropout_prob)
        self.pool_vector = nn.Parameter(torch.zeros(hidden_size))
        nn.init.xavier_uniform_(self.pool_vector.unsqueeze(0))
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, attention_mask=None):
        # (B, L, F)
        context, attn_probs = self.self_attention(x, attention_mask)
        context = self.layernorm(context)

        # 非线性变换 + 可学习聚合
        proj_context = torch.tanh(self.proj(context))
        score = (proj_context @ F.normalize(self.pool_vector, dim=0)) / (context.size(-1) ** 0.5)
        weight = F.softmax(score, dim=1).unsqueeze(-1)  # (B, L, 1)

        # 融合注意力
        pooled = torch.sum(context * weight, dim=1)
        pooled = pooled + context.mean(dim=1)  # 残差
        pooled = self.dropout(F.normalize(pooled, p=2, dim=-1))

        return pooled, attn_probs

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)  
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs_0 = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs_0)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs_0

class PMAReadout(nn.Module):
    def __init__(self, input_dim=64, num_seeds=2, output_dim=104, num_heads=4):
        super(PMAReadout, self).__init__()
        """
        Args:
            input_dim: 输入特征维度 (此处为 64)
            num_seeds: 想要输出的序列长度 (此处为 2)
            output_dim: 想要输出的特征维度 (此处为 104)
            num_heads: 多头注意力的头数
        """
        # 1. 定义可学习的种子向量 (Seeds)
        # 形状为 (1, num_seeds, output_dim)
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, output_dim))
        nn.init.xavier_uniform_(self.S)

        # 2. 定义多头注意力
        # 注意：为了直接输出 output_dim，我们将 k/v 投影到这个维度
        self.mha = nn.MultiheadAttention(embed_dim=output_dim, 
                                         num_heads=num_heads, 
                                         kdim=input_dim, 
                                         vdim=input_dim, 
                                         batch_first=True)
        
        # 3. 后处理层 (Feed Forward)
        self.layernorm = nn.LayerNorm(output_dim)
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        """
        x: (B, 30, 64)
        Returns: (B, 2, 104)
        """
        batch_size = x.size(0)
        
        # 将种子向量扩展到整个 Batch
        # (B, 2, 104)
        seeds = self.S.repeat(batch_size, 1, 1)
        
        # PMA 核心：Multi-head Attention
        # Query: seeds (B, 2, 104)
        # Key/Value: x (B, 30, 64)
        # 结果形状: (B, 2, 104)
        attn_output, _ = self.mha(seeds, x, x)
        
        # 残差连接与归一化
        out = self.layernorm(attn_output + seeds)
        out = out + F.relu(self.fc(out))
        
        return out

class Reduce1280to104(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 104),
            nn.LayerNorm(104)
        )

    def forward(self, x):
        """
        x: (B, 1280)
        return: (B, 104)
        """
        return self.net(x)


# iRMS-Net model
class iRMSNet(torch.nn.Module):
    def __init__(self,num_features_xd=78, n_head=4, num_features_xt=104, output_dim=128, dropout=0.2, n_gats=3):
        super(iRMSNet, self).__init__()

        '''
        num_features_xd: the feature dimension of drugs;
        n_head: the number of GATs attention heads;
        num_features_xt : the feature dimension of cancer cell lines;
        output_dim: the GATs hidden units, namely the embedding dimension of drug atoms;
        n_gats: the number of GATs layer, namely the range of substructures.
        '''

        # initial normal
        self.initial_norm = LayerNorm(num_features_xd)
        # graph drug convolution and drug layerNorm
        self.drug_gats = nn.ModuleList()
        self.drug_norm = nn.ModuleList()
        for i in range(n_gats):
            drug_gat = GAT_Block(n_head, num_features_xd, output_dim)
            self.add_module(f"drug_gat{i}", drug_gat)
            self.drug_gats.append(drug_gat)
            self.drug_norm.append(LayerNorm(output_dim * n_head))
            num_features_xd = output_dim * n_head
        self.drug_fc = nn.Linear(78,128)

        # drug interaction
        self.reduce=Reduce1280to104(dropout)
        self.CA=AttentionWrapper(4)
        self.drug=nn.Sequential(
            nn.Linear(512*3,512),#TODO
            nn.ReLU(),
            nn.Dropout(0.2)
        )



        # DL cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim * n_head),
            nn.ReLU()
        )

        self.readout=PMAReadout(128,2,104,4)
        # self.readout1=PMAReadout(64,1,104,4)

        self.cell=nn.Sequential(
            nn.Linear(num_features_xt,512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.cell1=nn.Sequential(
            nn.Linear(num_features_xt*2,512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.drugself=Drugself(feature_dim=512,num_heads=4,dropout_rate=0.2)
        self.cellself=Drugself(feature_dim=104,num_heads=4,dropout_rate=0.2)
        self.protein=Drugself(128,4,0.2)
        # self.protein1=Drugself(64,2,0.2)
        self.cocell=CellCellCoAttentionBlock(feature_dim=104,num_heads=4,dropout_rate=0.2)
        self.codrug=CellCellCoAttentionBlock(feature_dim=512,num_heads=4,dropout_rate=0.2)

    
        self.max = SelfAttentionPooling(hidden_size=512,num_attention_heads=4,dropout_prob=0.1)
        self.proteinmax=SelfAttentionPooling(hidden_size=104,num_attention_heads=4,dropout_prob=0.2)
        # self.tri_coatt = CellAwareDrugCrossAttention(dim=512, num_heads=4, cell_dim=104, dropout=0.1)


        self.cross = CrossAttention(query_dim=104, key_dim=512, num_heads=4,dropout=0.1)
        # self.cross1 = CrossAttention(query_dim=512, key_dim=104, num_heads=4,dropout=0.1)

        # combined layers
        #self.fc1 = nn.Linear(n_gats**2+2*n_gats, 128)
        self.fc1 = nn.Linear(512*3, 128)#TODO
        # self.fc2=nn.Linear(512,128)
        self.fc2 = nn.Linear(128, 2)
        # self.fc2 = nn.Linear(768, 128)
        # self.fc3 = nn.Linear(128, 2)

        # activation and regularization
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
        self.output_dim = output_dim

        self.cell2=nn.Linear(64,104)
    
    def save_num(self, d, path):
        d = d.cpu().numpy()
        ind = self.get_col_index(d)
        ind = pd.DataFrame(ind)
        ind.to_csv('data/case_study/' + path + '_index.csv', header=0, index=False)


    def forward(self, x1,x2, edge_index1,edge_index2, batch1,batch2, cell):
        # deal drug
        device = next(self.parameters()).device
        x1, x2 = x1.to(device), x2.to(device)
        edge_index1, edge_index2 = edge_index1.to(device), edge_index2.to(device)
        batch1, batch2 = batch1.to(device), batch2.to(device)
        cell = cell.to(device)

        repr_drug1 = []
        repr_drug2 = []
        x1 = self.initial_norm(x1, batch1)
        x2 = self.initial_norm(x2, batch2)
        for i, drug_gat in enumerate(self.drug_gats):
            drug1 = drug_gat(x1, edge_index1, batch1)
            drug2 = drug_gat(x2, edge_index2, batch2)
            h_1 = drug1[0]  # x
            r_1 = drug1[1]  # emb
            h_2 = drug2[0]  # x
            r_2 = drug2[1]  # emb
            repr_drug1.append(r_1)
            repr_drug2.append(r_2)
            h1 = self.drug_norm[i](h_1, batch1)
            h2 = self.drug_norm[i](h_2, batch2)
            x1 = self.elu(h1)
            x2 = self.elu(h2)


        repr_drug1 = torch.stack(repr_drug1, dim=-2)  # 【B，N，512】
        repr_drug2 = torch.stack(repr_drug2, dim=-2)  # [128,4,256]
        # druga0,_=self.max(repr_drug1)
        # drugb0,_=self.max(repr_drug2)

        # deal cell
        cell = F.normalize(cell, 2, 1)
        cell1=cell[:,:208]
        # cell3= cell[:, 104:1384]
        cell2= cell[:, 208:5968]
        # cell3=cell3.reshape(-1,20,64)
        cell2=cell2.reshape(-1,45,128)
        cell2=self.protein(cell2)
        # cell3=self.protein1(cell3)
        # cell2=cell2.max(dim=1).values #TODO 这是单独使用的
        # cell2=self.cell2(cell2)
        '''cell=torch.cat([cell1,cell2],dim=1)
        cell=self.CA(cell)#TODO'''
        # cell, cell_flat = self.CA(cell)#TODO #相似损失改动处
        cell=self.readout(cell2) 
        # cell3=self.readout1(cell3)
        # cell,_=self.proteinmax(cell)
        # cell3,_=self.proteinmax(cell3)
        # cell3=cell3.reshape(-1,104)
        # '''cell=cell2.reshape(-1,1,104)'''
        # #cell=cell.reshape(-1,1,104)
        cell=cell.reshape(-1,208)
        # cell = cell.reshape(cell.size(0), -1)
        # cell3=self.reduce(cell3)
        cell=torch.cat([cell,cell1],dim=1)
        cell=cell.reshape(-1,4,104)




        cell=self.CA(cell)

        cell=cell.reshape(-1,4,104)



        # fusion_out, feature_mask, x_orig = self.CA(cell)
        # cell = fusion_out
        #cell_vector = self.reduction(cell)
        ''' cell = cell.permute(0, 2, 1)'''


        cella,_=self.cross(cell,repr_drug1,repr_drug1)
        cellb,_=self.cross(cell,repr_drug2,repr_drug2)

        druga=self.drugself(repr_drug1)
        drugb=self.drugself(repr_drug2)

        cella=self.cellself(cella)
        cellb=self.cellself(cellb)

        cella,cellb=self.cocell(cella,cellb)

        cella=cella.max(dim=1).values
        cellb=cellb.max(dim=1).values


        cell_vector=torch.cat((cella,cellb),dim=1)

        cell_vector=self.cell1(cell_vector)

        # druga1,  tri_attn = self.tri_coatt(repr_drug1, repr_drug2, cell)
        # drugb1,  tri_attn = self.tri_coatt(repr_drug2, repr_drug1, cell)

        # druga1=druga.view(128,-1)
        # drugb1=drugb.view(128,-1)

        # druga1=self.drug(druga1)
        # drugb1=self.drug(drugb1)

        druga1,_=self.max(druga)
        drugb1,_=self.max(drugb)
        


        drugmix=torch.cat((druga1,drugb1),dim=1)
        

        xc = torch.cat((drugmix, cell_vector), 1)
        xc = F.normalize(xc, 2, 1)

        # # add some FC layers
        # xc = self.fc1(xc)
        # xc = self.elu(xc)
        # #xc = self.dropout(xc)
        # xc= self.fc2(xc)
        # out=xc
        # # out=out.squeeze(dim=1)
        # return out 
        
        # representation for margin separation改对比的地方 #TODO 
        feat = self.fc1(xc)
        feat = self.elu(feat)

        logits = self.fc2(feat)

        return logits, feat
        



import torch
import torch.nn as nn
import torch.nn.functional as F

class Drugself(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout_rate=0.1):
        """
        初始化完整的 Transformer Block
        :param feature_dim: F，特征维度（必须能被 num_heads 整除）
        :param num_heads: 多头注意力的头数 H
        :param dropout_rate: 用于防止过拟合的丢弃率
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # --- 1. 自注意力子层 (Self-Attention Sublayer) ---
        
        # 使用 PyTorch 内置高效的多头自注意力
        # batch_first=True 使得输入张量形状为 (B, L, F)
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 自注意力后的归一化
        self.norm1 = nn.LayerNorm(feature_dim)
        # 自注意力后的 Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # --- 2. 前馈网络子层 (Feed-Forward Sublayer) ---
        
        # 常见做法是中间维度是 feature_dim 的 4 倍
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
        # 前馈网络后的归一化
        self.norm2 = nn.LayerNorm(feature_dim)
        # 前馈网络后的 Dropout
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x 形状: (B, L, F)
        
        # ====================
        # 1. 自注意力子层
        # ====================
        
        # 1a. 计算注意力输出
        # attn_output 是 (B, L, F)
        # attn_weights 是 (B, L, L)，通常不需要，但 PyTorch 也会返回
        attn_output, _ = self.attn(
            query=x,
            key=x,
            value=x
        )
        
        # 1b. 残差连接 + Dropout + 层归一化
        # 公式：x = LayerNorm(x + Dropout(Attention(x)))
        x = x + self.dropout1(attn_output)  # 残差连接
        x = self.norm1(x)                   # 层归一化
        
        # ====================
        # 2. 前馈网络子层
        # ====================
        
        # 2a. 计算 FFN 输出
        ffn_output = self.ffn(x)
        
        # 2b. 残差连接 + Dropout + 层归一化
        # 公式：x = LayerNorm(x + Dropout(FFN(x)))
        x = x + self.dropout2(ffn_output) # 残差连接
        x = self.norm2(x)                 # 层归一化
        
        # 输出 x 形状仍然是 (B, L, F)，是更新后的药物特征张量
        return x
    

class CrossAttentionEncoderBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout_rate=0.1, ffn_expansion=2):
        """
        完整的交叉注意力编码器块：注意力 + FFN
        :param feature_dim: F，特征维度
        :param num_heads: 多头注意力的头数 H
        :param dropout_rate: Dropout 丢弃率
        :param ffn_expansion: FFN 中间层的维度扩张倍数 (通常是 4)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # --- 1. 交叉注意力子层 (Cross-Attention Sublayer) ---
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # --- 2. 前馈网络子层 (Feed-Forward Network Sublayer) ---
        
        # FFN: Linear -> ReLU -> Linear
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * ffn_expansion),
            nn.ReLU(),
            nn.Linear(feature_dim * ffn_expansion, feature_dim)
        )
        
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout2 = nn.Dropout(dropout_rate)


    def forward(self, query_tensor, kv_tensor):
        """
        执行交叉注意力 (Query 关注 Key/Value) 并进行 FFN 处理
        :param query_tensor: 作为 Q 的输入 (B, L_Q, F)
        :param kv_tensor: 作为 K 和 V 的输入 (B, L_KV, F)
        :return: 经过双重更新的 query_tensor (B, L_Q, F)
        """
        
        # --- 1. 交叉注意力 (带残差和归一化) ---
        
        # Q = query_tensor, K = kv_tensor, V = kv_tensor
        attn_output, _ = self.cross_attn(
            query=query_tensor,
            key=kv_tensor,
            value=kv_tensor
        )
        
        # 第一次残差连接 + 归一化
        # x = LayerNorm(Q + Dropout(Attn(Q, K, V)))
        x = query_tensor + self.dropout1(attn_output) # 残差连接加在 Q 上
        x = self.norm1(x)                             # LayerNorm
        
        
        # --- 2. 前馈网络 (带残差和归一化) ---
        
        ffn_output = self.ffn(x)
        
        # 第二次残差连接 + 归一化
        # x = LayerNorm(x + Dropout(FFN(x)))
        x = x + self.dropout2(ffn_output) # 残差连接
        x = self.norm2(x)                 # LayerNorm
        
        return x


class CellCellCoAttentionBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout_rate=0.1):
        """
        双向细胞共注意力块 (包含 FFN)
        """
        super().__init__()
        
        # Cell A 关注 Cell B (更新 A 的特征)
        self.attn_A_to_B = CrossAttentionEncoderBlock(feature_dim, num_heads, dropout_rate)
        
        # Cell B 关注 Cell A (更新 B 的特征)
        self.attn_B_to_A = CrossAttentionEncoderBlock(feature_dim, num_heads, dropout_rate)
        
    def forward(self, cell_A_features, cell_B_features):
        
        # 1. A 关注 B (A 使用 B 的上下文信息更新自己)
        updated_A = self.attn_A_to_B(
            query_tensor=cell_A_features,
            kv_tensor=cell_B_features
        )
        
        # 2. B 关注 A (B 使用 A 的上下文信息更新自己)
        updated_B = self.attn_B_to_A(
            query_tensor=cell_B_features,
            kv_tensor=cell_A_features
        )
        
        # 返回经过交叉注意力和 FFN 增强的细胞特征
        return updated_A, updated_B
