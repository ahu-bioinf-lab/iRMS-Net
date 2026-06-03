
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from model.iRMSNet import *
from utils_test import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
from torch_geometric.loader import DataLoader
import datetime


# 定义batch hard margin loss 对比增加处 #TODO
def batch_hard_margin_sep_loss(emb, y, margin=0.5):
    """
    emb: [B, D]   (fc1后的embedding)
    y:   [B]      (0/1 labels)
    """
    y = y.view(-1)
    pos_mask = (y == 1)
    neg_mask = (y == 0)

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return emb.new_tensor(0.0)

    pos_emb = emb[pos_mask]
    neg_emb = emb[neg_mask]

    # normalize for stability
    pos_emb = F.normalize(pos_emb, dim=1)
    neg_emb = F.normalize(neg_emb, dim=1)

    # pairwise distances [P, N]
    dist = torch.cdist(pos_emb, neg_emb, p=2)

    # hardest negative for each positive
    hardest_neg_dist = dist.min(dim=1).values

    loss = F.relu(margin - hardest_neg_dist).mean()
    return loss


# ======================================================
# 🔒 1. 固定随机种子（可复现的基础）
# ======================================================
SEED = 520
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # cuDNN 中的一些卷积默认是非确定性的，这两句可强制确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================
# 2. 训练函数
# ======================================================
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()

    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        # zip 两个 loader 必须顺序完全一致，因此 loader 不能 shuffle
        data1 = data[0]
        data2 = data[1]

        x1, edge_index1, batch1, cell = (
            data1.x.to(device),
            data1.edge_index.to(device),
            data1.batch.to(device),
            data1.cell.to(device)
        )
        x2, edge_index2, batch2 = (
            data2.x.to(device),
            data2.edge_index.to(device),
            data2.batch.to(device)
        )

        y = data1.y.view(-1).long().to(device)

        optimizer.zero_grad()

# #定义损失函数修改处 #TODO
#         output = model(x1, x2, edge_index1, edge_index2, batch1, batch2, cell)
#         #output, aux = model(x1, x2, edge_index1, edge_index2, batch1, batch2, cell)
#         loss = loss_fn(output, y)
#         loss.backward()
        
        output, feat = model(x1, x2, edge_index1, edge_index2, batch1, batch2, cell)

        loss_cls = loss_fn(output, y)

        loss_sep = batch_hard_margin_sep_loss(
            feat,
            y,
            margin=0.6
            )

        lambda_sep = 0.1  # 推荐起点
        loss = loss_cls + lambda_sep * loss_sep
        loss.backward() #TODO


    #     loss_main = loss_fn(output, y)

    #     loss_inf = 0.0
    #     if aux is not None and aux.get("mask_out", None) is not None:
    #         ratio = 0.25 * epoch / NUM_EPOCHS
    #         loss_inf = model.infmask_loss_fn(
    #         aux["q"],
    #         aux["mask_out"],
    #         ratio
    # )

    #     loss = loss_main + 0.1 * loss_inf

    #     loss.backward()


        
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(drug1_loader_train),
                100. * batch_idx / len(drug1_loader_train), loss.item()))


# ======================================================
# 3. 预测函数
# ======================================================
def predicting(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))

    with torch.no_grad():
        for data1, data2 in zip(drug1_loader_test, drug2_loader_test):

            x1, edge_index1, batch1, cell = (
                data1.x.to(device),
                data1.edge_index.to(device),
                data1.batch.to(device),
                data1.cell.to(device)
            )
            x2, edge_index2, batch2 = (
                data2.x.to(device),
                data2.edge_index.to(device),
                data2.batch.to(device)
            )

            # output = model(x1, x2, edge_index1, edge_index2, batch1, batch2, cell)
            output, _ = model(x1, x2, edge_index1, edge_index2, batch1, batch2, cell)#TODO 



            ys = F.softmax(output, 1).cpu().numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]

            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


# ======================================================
# 4. 全局参数
# ======================================================
modeling = iRMSNet
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.00003
LOG_INTERVAL = 20
NUM_EPOCHS = 500
FOLD_NUM = 5
datafile = 'comb-2'
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = DATA_DIR / 'iRMS-Net'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
print('Cross Validation Folds: ', FOLD_NUM)

# ======================================================
# 5. 设备选择
# ======================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# ======================================================
# 6. 数据加载
# ======================================================
drug1_data = TestbedDataset(root=str(DATA_DIR), dataset=datafile + '_drug1')
drug2_data = TestbedDataset(root=str(DATA_DIR), dataset=datafile + '_drug2')

lenth = len(drug1_data)
pot = int(lenth / FOLD_NUM)
print('Total samples:', lenth)
print('Samples per fold:', pot)

assert len(drug1_data) == len(drug2_data)


# ======================================================
# 🔥 7. 一次性全局 shuffle（非常关键）
#    - 保证 drug1 和 drug2 顺序完全一致
#    - loader 不再 shuffle，也能保证随机性
#    - 完全可复现
# ======================================================
indices = np.arange(lenth)
np.random.seed(SEED)
np.random.shuffle(indices)

drug1_data = drug1_data[indices]
drug2_data = drug2_data[indices]

print("Dataset shuffled ONCE with fixed seed.")


# ======================================================
# 8. 五折循环准备
# ======================================================
fold_metrics = []
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
summary_file = OUTPUT_DIR / f'iRMSNet_Summary_{LR}_{NUM_EPOCHS}_Adam_{timestamp}.txt'

with open(summary_file, 'w') as f:
    f.write('Fold\tAUC\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL\n')


# ======================================================
# 9. 五折交叉验证（划分后不 shuffle）
# ======================================================
# 依然沿用你原本 random_num 的五折逻辑，只是数据已提前 shuffle
random_num = list(range(lenth))  # 此时已与 drug1/drug2 对齐


for i in range(FOLD_NUM):
    print(f"\n==================== Fold {i+1}/{FOLD_NUM} ====================")

    test_num = random_num[pot * i : pot * (i+1)]
    train_num = random_num[:pot * i] + random_num[pot * (i+1):]

    drug1_train = [drug1_data[j] for j in train_num]
    drug2_train = [drug2_data[j] for j in train_num]
    drug1_test = [drug1_data[j] for j in test_num]
    drug2_test = [drug2_data[j] for j in test_num]

    # ======================================================
    # ❗ 关键改动：DataLoader 不能 shuffle，否则 zip 会错乱
    #    num_workers=0 → 完全避免多进程带来的非确定性
    # ======================================================
    drug1_loader_train = DataLoader(drug1_train, batch_size=TRAIN_BATCH_SIZE,
                                    shuffle=False, num_workers=0)
    drug1_loader_test = DataLoader(drug1_test, batch_size=TEST_BATCH_SIZE,
                                   shuffle=False, num_workers=0)

    drug2_loader_train = DataLoader(drug2_train, batch_size=TRAIN_BATCH_SIZE,
                                    shuffle=False, num_workers=0)
    drug2_loader_test = DataLoader(drug2_test, batch_size=TEST_BATCH_SIZE,
                                   shuffle=False, num_workers=0)

    # 模型与优化器
    model = modeling().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 文件保存路径
    model_file_name = OUTPUT_DIR / f'iRMSNet_Fold{i+1}_{timestamp}.model'
    fold_metric_file = OUTPUT_DIR / f'iRMSNet_Fold{i+1}_{LR}_{NUM_EPOCHS}_Adam_{timestamp}.txt'

    with open(fold_metric_file, 'w') as f:
        f.write('Epoch\tAUC\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL\n')

    best_auc = -1.0
    best_metrics = None

    # ======================================================
    # 10. 每折训练 epoch
    # ======================================================
    for epoch in range(NUM_EPOCHS):
        train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch+1)

        T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)

        AUC = roc_auc_score(T, S)
        precision_curve, recall_curve, _ = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall_curve, precision_curve)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        PREC = precision_score(T, Y) if (tp + fp) != 0 else 0.0
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        RECALL = recall_score(T, Y)
        


        with open(fold_metric_file, 'a') as f:
            f.write(f'{epoch+1}\t{AUC:.6f}\t{PR_AUC:.6f}\t{ACC:.6f}\t{BACC:.6f}\t'
                    f'{PREC:.6f}\t{TPR:.6f}\t{KAPPA:.6f}\t{RECALL:.6f}\n')

        if AUC > best_auc:
            best_auc = AUC
            best_metrics = [i+1, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, RECALL]

            torch.save(model.state_dict(), model_file_name)
            save_AUCs([epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, RECALL], fold_metric_file)

        print(f'Epoch {epoch+1} | AUC:{AUC:.3f}, PR_AUC:{PR_AUC:.3f}, RECALL:{RECALL:.3f}')

    fold_metrics.append(best_metrics[1:])

    with open(summary_file, 'a') as f:
        f.write('\t'.join([str(best_metrics[0])] +
                          [f'{x:.6f}' for x in best_metrics[1:]]) + '\n')

    print(f"\nFold {i+1} Best Result | AUC:{best_metrics[1]:.3f}, PR_AUC:{best_metrics[2]:.3f}, RECALL:{best_metrics[8]:.3f}")


# ======================================================
# 11. 五折均值 ± 标准差
# ======================================================
print("\n" + "="*80)
print("                      5-Fold Cross Validation Summary")
print("="*80)

fold_metrics_np = np.array(fold_metrics)
metrics_mean = np.mean(fold_metrics_np, axis=0)
metrics_std = np.std(fold_metrics_np, axis=0)

metric_names = ['AUC', 'PR_AUC', 'ACC', 'BACC', 'PREC', 'TPR', 'KAPPA', 'RECALL']

for name, mean_val, std_val in zip(metric_names, metrics_mean, metrics_std):
    print(f"{name:8s}: {mean_val:.4f} ± {std_val:.4f}")

with open(summary_file, 'a') as f:
    f.write('\n' + '='*50 + '\n')
    f.write('Mean±Std\t' + '\t'.join([f'{m:.6f}±{s:.6f}' for m, s in zip(metrics_mean, metrics_std)]) + '\n')

print("="*80)
print(f"Summary results saved to: {summary_file}")
