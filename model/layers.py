import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import (GATConv,SAGPooling,global_add_pool)
import math


class GAT_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.conv = GATConv(in_features, head_out_feats, n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)

    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index)
        att_x, att_edge_index, att_edge_attr, att_batch, att_perm, att_scores = self.readout(x, edge_index, batch=batch)
        global_graph_emb = global_add_pool(att_x, att_batch)
        return x, global_graph_emb


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
#   TODO  B,L,F  b,n,f  B,F

#融合GAT各层表示为一层
class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=4, dropout_prob=0.1):
        super(SelfAttentionPooling, self).__init__()
        self.self_attention = SelfAttention(hidden_size, num_attention_heads, dropout_prob)
        # 用一个可学习的向量做最后加权融合
        self.pool_vector = nn.Parameter(torch.zeros(hidden_size))
        nn.init.xavier_uniform_(self.pool_vector.unsqueeze(0))

    def forward(self, x, attention_mask=None):
        """
        x: (B, L=6, F)
        attention_mask: (B, 1, 1, L) or None
        """
        # 1. 自注意力
        context, attn_probs = self.self_attention(x, attention_mask)  # (B, L, F)

        # 2. 融合：用可学习向量做加权
        # pool_vector: (F,) -> (1, F, 1) 用作矩阵乘加权
        weight = F.softmax(context @ self.pool_vector, dim=1)  # (B, L)
        weight = weight.unsqueeze(-1)  # (B, L, 1)

        # 3. 加权求和得到最终融合向量
        pooled = torch.sum(context * weight, dim=1)  # (B, F)
        return pooled, attn_probs
    

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

class CellAwareDrugCrossAttention(nn.Module):
    """
    Cell-Aware Cross-Attention with Head-Aware Additive Bias.
    Bias is different for each attention head, but constant across sequence positions.
    """
    def __init__(self, dim, cell_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.d_head = dim // num_heads
        
        # Projections (Q, K, V remains the same)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # --- MODIFICATION: Cell Projection for Head-Aware Bias ---
        # 1. 减少 cell_dim 到 dim (可选，保持与模块一/二一致)
        self.cell_reduce = nn.Linear(cell_dim, dim)
        # 2. 投影到与头数 H 相同的维度，每个头一个偏差值
        self.cell_proj_for_head_bias = nn.Linear(dim, self.num_heads)
        
        # learnable bias weight
        self.lambda_param = nn.Parameter(torch.tensor(0.5))
        
        # normalization & dropout
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # small FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, drug1, drug2, cell, attn_mask=None):
        B, L1, D = drug1.shape
        _, L2, _ = drug2.shape
        
        # ---- base Q, K, V ----
        Q = self.q_proj(drug1).view(B, L1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        K = self.k_proj(drug2).view(B, L2, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.v_proj(drug2).view(B, L2, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, L1, L2)
        
        # --- MODIFICATION: Head-Aware Cell Bias Calculation ---
        if cell.dim() == 3:
            cell_summary = cell.mean(dim=1)  # (B, Dc)
        else:
            cell_summary = cell  # (B, Dc)
        
        cell_vec = self.cell_reduce(cell_summary)  # (B, D)
        
        # 投影到 H 维，得到每个头的偏差 (B, H)
        head_bias_vec = self.cell_proj_for_head_bias(cell_vec)  # (B, H)
        
        lam = torch.sigmoid(self.lambda_param)
        
        # 扩展维度以广播: (B, H) -> (B, H, 1, 1)
        # 偏差现在对每个头是不同的。
        head_bias = lam * head_bias_vec.unsqueeze(-1).unsqueeze(-1)
        
        logits = logits + head_bias
        
        # ---- attention ----
        if attn_mask is not None:
            logits = logits + attn_mask
        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        
        # ---- weighted sum ----
        out = torch.matmul(attn, V)  # (B, H, L1, d)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L1, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # ---- residual + FFN ----
        out = self.norm(drug1 + out)
        out = out + self.ffn(out)
        
        return out, attn.detach()

class DrugDrugCoAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
    
    def forward(self, drugA, drugB):
        # drugA, drugB: [B, L, F]
        # A attends to B
        A2B, _ = self.attn(query=drugA, key=drugB, value=drugB)
        # B attends to A
        B2A, _ = self.attn(query=drugB, key=drugA, value=drugA)
        return A2B, B2A
