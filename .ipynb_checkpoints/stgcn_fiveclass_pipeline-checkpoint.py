#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ST-GCN 五分类（walk/lie/stand/sit/bend）完整管线，一文件版
================================================================
本脚本把你上面的所有步骤整合到一起：
1) 标签统一到 5 类（含别名映射）
2) 用 kept_idx（去零帧后的保留索引）对齐标签
3) 窗口化成 ST-GCN 输入形状 (N,3,T,32,1)
4) 生成 train/val/test 三个 NPZ（推荐：16GZ→train, 17JP→val, 19MM→test）
5) ST-GCN 原版图划分（root/centripetal/centrifugal），基于 constants.py 的 Kinect-32 joints
6) Dataset / DataLoader
7) 训练 + 验证 + 最终测试（打印整体准确率 & 混淆矩阵）

使用方式（Notebook 或命令行都可）：
------------------------------------------------
# 先修改下面 DEFAULT_PATHS 里的标签路径（逐帧标签，与原始 T 对齐）

# 仅打包 train/val/test：
python stgcn_fiveclass_pipeline.py pack --win 120 --stride 30 --strict --mode center

# 训练+验证+测试：
python stgcn_fiveclass_pipeline.py train --epochs 30 --batch_size 32 --lr 1e-3

# 或在 Notebook：
from stgcn_fiveclass_pipeline import pack_train_val_test, train_main
pack_train_val_test(win=120, stride=30, strict=True, mode="center")
train_main(epochs=30, batch_size=32, lr=1e-3)

依赖：
- constants.py （需与本脚本同目录，提供 KINECT_JOINT_NAMES / KINECT_LIMB_CONNECTIONS / KINECT_KPT_DIMS）
- Numpy, PyTorch, scikit-learn（用于混淆矩阵）
"""

from __future__ import annotations
import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import confusion_matrix, classification_report
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# ======== 0) 路径与五类配置 ========
@dataclass
class DefaultPaths:
    # 预处理后的姿态（已 center+scale+去零帧）
    pose_16gz: str = "data/poses_16GZ_cs_nz.npy"
    pose_17jp: str = "data/poses_17JP_cs_nz.npy"
    pose_19mm: str = "data/poses_19MM_cs_nz.npy"
    # 保留索引（与上面姿态对应）
    keep_16gz: str = "data/poses_16GZ_kept_idx.npy"
    keep_17jp: str = "data/poses_17JP_kept_idx.npy"
    keep_19mm: str = "data/poses_19MM_kept_idx.npy"
    # 原始逐帧标签（与原始 T 对齐）——👉 你需要改成自己的路径
    lab_16gz: str = "data/labels_16GZ.npy"
    lab_17jp: str = "data/labels_17JP.npy"
    lab_19mm: str = "data/labels_19MM.npy"
    # 打包后的三份数据
    out_train: str = "data/train_stgcn.npz"
    out_val:   str = "data/val_stgcn.npz"
    out_test:  str = "data/test_stgcn.npz"

DEFAULT_PATHS = DefaultPaths()

FIVE_LABELS = ["walk", "lie", "stand", "sit", "bend"]
FIVE_SET = set(FIVE_LABELS)

# 常见别名映射（可按需继续补充）
ALIAS = {
    # walk
    "walking": "walk", "walks": "walk", "走": "walk",
    # lie
    "lying": "lie", "p_lie": "lie", "卧": "lie", "lay": "lie",
    # stand
    "standing": "stand", "站": "stand",
    # sit
    "sitting": "sit", "p_sit": "sit", "坐": "sit",
    # bend
    "bending": "bend", "弯腰": "bend", "stoop": "bend",
}

# ======== 1) 标签统一 ========
def map_to_five(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    s = str(label).strip().lower()
    s = ALIAS.get(s, s)
    return s if s in FIVE_SET else None


def align_labels_to_kept_idx(label_path: str, kept_idx_path: str) -> np.ndarray:
    """将原始逐帧标签对齐到去零帧后的 K 长度，并做五类映射。
    返回 (K,)，元素是五类名或 None。
    """
    y = np.load(label_path, allow_pickle=True)  # (T,)
    keep = np.load(kept_idx_path)               # (K,)
    yk = y[keep]
    yk = np.array([map_to_five(v) for v in yk], dtype=object)
    return yk


# ======== 2) 窗口化（严格只留五类窗口） ========
from collections import Counter

def window_label(window_labels: List[Optional[str]], mode: str = "center") -> Optional[str]:
    if mode == "center":
        return window_labels[len(window_labels)//2]
    # majority
    valid = [v for v in window_labels if v is not None]
    if not valid:
        return None
    c = Counter(valid)
    return c.most_common(1)[0][0]


def make_windows_stgcn_from_paths(x_path: str, label_path: str, kept_idx_path: str,
                                  win: int = 120, stride: int = 30,
                                  strict: bool = True, mode: str = "center") -> Tuple[np.ndarray, np.ndarray]:
    """
    x_path: (K,32,3)  已经 center+scale+去零 的序列
    label_path: 原始逐帧标签 (T,)
    kept_idx_path: 对齐索引 (K,)
    返回：X:(N,3,T,32,1), Y:(N,)（int 类别 id）
    """
    Xseq = np.load(x_path)                         # (K,32,3)
    Yseq = align_labels_to_kept_idx(label_path, kept_idx_path)  # (K,)
    K = len(Xseq)

    xs, ys = [], []
    cls2id = {c: i for i, c in enumerate(FIVE_LABELS)}

    for s in range(0, K - win + 1, stride):
        ly = Yseq[s:s+win]
        if strict and any(v is None for v in ly):
            continue
        ywin = window_label(ly if strict else [v for v in ly if v is not None], mode=mode)
        if ywin is None or ywin not in FIVE_SET:
            continue
        seg = Xseq[s:s+win]                 # (T,32,3)
        seg = np.transpose(seg, (2, 0, 1))  # -> (3,T,32)
        seg = seg[..., None]                # -> (3,T,32,1)
        xs.append(seg)
        ys.append(cls2id[ywin])

    if not xs:
        return np.empty((0, 3, win, 32, 1)), np.empty((0,), dtype=np.int64)

    X = np.stack(xs)
    Y = np.array(ys, dtype=np.int64)
    return X, Y


# ======== 3) 打包 train/val/test ========
def pack_train_val_test(win: int = 120, stride: int = 30, strict: bool = True, mode: str = "center",
                        paths: DefaultPaths = DEFAULT_PATHS) -> Tuple[str, str, str]:
    os.makedirs(os.path.dirname(paths.out_train), exist_ok=True)

    # Train = 16GZ
    Xtr, Ytr = make_windows_stgcn_from_paths(paths.pose_16gz, paths.lab_16gz, paths.keep_16gz,
                                             win=win, stride=stride, strict=strict, mode=mode)
    np.savez_compressed(paths.out_train, X=Xtr, Y=Ytr, meta=dict(labels=FIVE_LABELS))
    print("[pack] train:", Xtr.shape, Ytr.shape)

    # Val = 17JP
    Xva, Yva = make_windows_stgcn_from_paths(paths.pose_17jp, paths.lab_17jp, paths.keep_17jp,
                                             win=win, stride=stride, strict=strict, mode=mode)
    np.savez_compressed(paths.out_val, X=Xva, Y=Yva, meta=dict(labels=FIVE_LABELS))
    print("[pack] val  :", Xva.shape, Yva.shape)

    # Test = 19MM
    Xte, Yte = make_windows_stgcn_from_paths(paths.pose_19mm, paths.lab_19mm, paths.keep_19mm,
                                             win=win, stride=stride, strict=strict, mode=mode)
    np.savez_compressed(paths.out_test, X=Xte, Y=Yte, meta=dict(labels=FIVE_LABELS))
    print("[pack] test :", Xte.shape, Yte.shape)

    return paths.out_train, paths.out_val, paths.out_test


# ======== 4) Graph（原版 ST-GCN spatial partition） ========
from constants import KINECT_JOINT_NAMES, KINECT_LIMB_CONNECTIONS

J2I = {name: idx for idx, name in enumerate(KINECT_JOINT_NAMES)}
EDGES_32 = [(J2I[a], J2I[b]) for a, b in KINECT_LIMB_CONNECTIONS]
CENTER = J2I.get('pelvis', 0)


def edge2adj(V: int, edges: List[Tuple[int, int]]):
    A = np.zeros((V, V), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A


def normalize_digraph(A: np.ndarray) -> np.ndarray:
    Dl = np.sum(A, axis=0)
    Dl[Dl == 0] = 1
    Dn = np.diag(1.0 / Dl)
    return A @ Dn


def hop_distance(V: int, edges: List[Tuple[int, int]], max_hop: int = 2) -> np.ndarray:
    A = edge2adj(V, edges)
    hop_dis = np.full((V, V), np.inf)
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop + 1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:
    def __init__(self, num_node: int = 32, edges: List[Tuple[int, int]] = EDGES_32,
                 center: int = CENTER, max_hop: int = 2, dilation: int = 1):
        self.num_node = num_node
        self.A = self.get_adjacency(num_node, edges, center, max_hop, dilation)

    def get_adjacency(self, V, edges, center, max_hop, dilation):
        hop_dis = hop_distance(V, edges, max_hop)
        valid_hop = range(0, max_hop + 1, dilation)

        A_root, A_close, A_far = np.zeros((V, V)), np.zeros((V, V)), np.zeros((V, V))
        A = edge2adj(V, edges)

        for i in range(V):
            for j in range(V):
                if hop_dis[j, i] in valid_hop and A[j, i] > 0:
                    if hop_dis[j, center] == hop_dis[i, center]:
                        A_root[j, i] = A[j, i]
                    elif hop_dis[j, center] < hop_dis[i, center]:
                        A_close[j, i] = A[j, i]
                    else:
                        A_far[j, i] = A[j, i]

        A_root = normalize_digraph(A_root)
        A_close = normalize_digraph(A_close)
        A_far = normalize_digraph(A_far)
        return np.stack([A_root, A_close, A_far], axis=0).astype(np.float32)  # (3,V,V)


# ======== 5) Dataset / Dataloader ========
class STGCNWindowDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"]          # (N,3,T,V,1)
        self.Y = data["Y"]          # (N,)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).float().squeeze(-1)  # (3,T,V)
        y = torch.tensor(int(self.Y[i])).long()
        return x, y


# ======== 6) 原版风格 ST-GCN（最小可用实现） ========
class GCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray, coff_embedding: int = 4, num_subset: int = 3):
        super().__init__()
        inter_channels = out_channels // coff_embedding
        self.num_subset = num_subset
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))  # (3,V,V)

        self.conv_a = nn.ModuleList([nn.Conv2d(in_channels, inter_channels, 1) for _ in range(num_subset)])
        self.conv_b = nn.ModuleList([nn.Conv2d(in_channels, inter_channels, 1) for _ in range(num_subset)])
        self.conv_d = nn.ModuleList([nn.Conv2d(inter_channels, out_channels, 1) for _ in range(num_subset)])

        self.bn = nn.BatchNorm2d(out_channels)
        self.down = nn.Identity() if (in_channels == out_channels) else nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x: (N,C,T,V)
        N, C, T, V = x.shape
        y = None
        A = self.A
        for i in range(self.num_subset):
            # 这里用固定 A 的聚合（最小实现）；可改成注意力可学习 A。
            z = torch.einsum('nctv,vw->nctw', x, A[i])  # (N,C,T,V)
            z = self.conv_d[i](z)
            y = z if y is None else y + z
        y = self.bn(y)
        y = y + self.down(x)
        return self.relu(y)


class TCN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=(9, 1), padding=(4, 0))
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class STGCN(nn.Module):
    def __init__(self, in_channels: int, num_class: int, A: np.ndarray):
        super().__init__()
        V = A.shape[-1]
        self.data_bn = nn.BatchNorm1d(in_channels * V)

        self.g1 = GCN(in_channels, 64, A);  self.t1 = TCN(64)
        self.g2 = GCN(64, 64, A);          self.t2 = TCN(64)
        self.g3 = GCN(64,128, A);          self.t3 = TCN(128)
        self.g4 = GCN(128,256, A);         self.t4 = TCN(256)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(256, num_class)
        )

    def forward(self, x):  # x: (N,3,T,32)
        N, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(N, T, C * V)
        x = self.data_bn(x.transpose(1, 2)).transpose(1, 2)
        x = x.view(N, T, C, V).permute(0, 2, 1, 3).contiguous()

        x = self.t1(self.g1(x))
        x = self.t2(self.g2(x))
        x = self.t3(self.g3(x))
        x = self.t4(self.g4(x))
        return self.head(x)


# ======== 7) 训练 / 验证 / 测试 ========
@dataclass
class TrainConfig:
    train_npz: str = DEFAULT_PATHS.out_train
    val_npz: str = DEFAULT_PATHS.out_val
    test_npz: str = DEFAULT_PATHS.out_test
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    device: Optional[str] = None


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    correct, total = 0, 0
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            all_y.append(y.cpu().numpy())
            all_p.append(pred.cpu().numpy())
    acc = correct / max(total, 1)
    y_true = np.concatenate(all_y) if all_y else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_p) if all_p else np.array([], dtype=np.int64)
    return acc, y_true, y_pred


def train_main(epochs: int = 30, batch_size: int = 32, lr: float = 1e-3,
               train_npz: str = DEFAULT_PATHS.out_train,
               val_npz: str = DEFAULT_PATHS.out_val,
               test_npz: str = DEFAULT_PATHS.out_test,
               device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Graph
    G = Graph()
    A = G.A  # (3,32,32)

    # Data
    train_ds = STGCNWindowDataset(train_npz)
    val_ds   = STGCNWindowDataset(val_npz)
    test_ds  = STGCNWindowDataset(test_npz)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = STGCN(in_channels=3, num_class=len(FIVE_LABELS), A=A).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None

    for ep in range(1, epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()
            run_loss += float(loss.item()) * x.size(0)
            seen += int(x.size(0))

        val_acc, _, _ = evaluate(model, val_loader, device)
        print(f"[Ep {ep:03d}] loss={run_loss/max(seen,1):.4f} | val_acc={val_acc*100:.2f}%")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # 测试：用最佳验证集权重
    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"[TEST] acc={test_acc*100:.2f}%  on {len(y_true)} samples")
    if _HAS_SK and len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(FIVE_LABELS))))
        print("Confusion Matrix (rows=true, cols=pred):\n", cm)
        print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=FIVE_LABELS, digits=3))


# ======== 8) CLI ========
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pack = sub.add_parser("pack", help="仅打包 train/val/test 三份 npz")
    p_pack.add_argument("--win", type=int, default=120)
    p_pack.add_argument("--stride", type=int, default=30)
    p_pack.add_argument("--strict", action="store_true")
    p_pack.add_argument("--mode", type=str, default="center", choices=["center", "majority"])

    p_train = sub.add_parser("train", help="训练+验证+测试")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)

    args = p.parse_args()

    if args.cmd == "pack":
        pack_train_val_test(win=args.win, stride=args.stride, strict=args.strict, mode=args.mode)
    elif args.cmd == "train":
        train_main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
