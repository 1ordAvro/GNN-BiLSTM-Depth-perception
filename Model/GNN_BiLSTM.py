import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# CHANNELS
# -------------------------------
SELECTED_CHANNELS = [
    "Fp1-A1","Fp2-A2","F3-A1","F4-A2",
    "C3-A1","C4-A2","P3-A1","P4-A2",
    "O1-A1","O2-A2","F7-A1","F8-A2",
    "T3-A1","T4-A2","T5-A1","T6-A2"
]

CHANNEL_LABELS = [ch.split('-')[0] for ch in SELECTED_CHANNELS]

np.random.seed(42)
FIXED_POS = {i: np.random.rand(2) for i in range(len(CHANNEL_LABELS))}

# -------------------------------
# DEVICE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# SPECTRAL WEIGHTING
# -------------------------------
class SpectralWeighting(nn.Module):
    def __init__(self, C, F):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(C, F))

    def forward(self, x):
        w = torch.sigmoid(self.weights)
        return x * w.unsqueeze(0).unsqueeze(0)

# -------------------------------
# GRAPH
# -------------------------------
def compute_graph(x):
    A_list = []
    for t in range(x.shape[0]):
        H = x[t]
        H = H / (H.norm(dim=1, keepdim=True) + 1e-8)
        A_list.append(torch.matmul(H, H.T))

    A = torch.stack(A_list).mean(0)
    A = (A + A.T) / 2
    return A

def sparsify_graph(A, k=2, threshold=0.3):
    C = A.shape[0]
    A_sparse = torch.zeros_like(A)

    for i in range(C):
        vals, idx = torch.topk(A[i], k + 1)
        for j in idx:
            if i != j and A[i, j] > threshold:
                A_sparse[i, j] = A[i, j]
                A_sparse[j, i] = A[i, j]

    return A_sparse

# -------------------------------
# GAT
# -------------------------------
class MultiHeadGAT(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a = nn.Parameter(torch.randn(num_heads, 2 * out_features))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(out_features * num_heads)
        self.residual = nn.Linear(in_features, out_features * num_heads)

        self.temperature = 1.5

    def forward(self, H, A):
        B, C, _ = H.shape

        Wh = self.W(H).view(B, C, self.num_heads, self.out_features)

        Wh_i = Wh.unsqueeze(2).repeat(1, 1, C, 1, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, C, 1, 1, 1)

        e = torch.cat([Wh_i, Wh_j], dim=-1)
        e = torch.einsum("bijhf,hf->bijh", e, self.a)
        e = self.leakyrelu(e)

        mask = (A > 0)
        eye = torch.eye(C, device=A.device).bool()
        mask = mask | eye.unsqueeze(0).expand_as(mask)

        e = e.masked_fill(~mask.unsqueeze(-1), -1e9)

        A_norm = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        e = e + A_norm.unsqueeze(-1)

        alpha = torch.softmax(e / self.temperature, dim=2)
        alpha = self.dropout(alpha)

        H_out = torch.einsum("bijh,bjhf->bihf", alpha, Wh)
        H_out = H_out.reshape(B, C, -1)

        H_out = self.norm(H_out + self.residual(H))

        return F.relu(H_out), alpha

# -------------------------------
# CHANNEL ATTENTION
# -------------------------------
class ChannelAttentionPooling(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.attn = nn.Linear(F, 1)

    def forward(self, x):
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)

# -------------------------------
# TEMPORAL ATTENTION
# -------------------------------
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.temperature = 1.5

    def forward(self, x):
        scores = torch.tanh(self.W(x))
        scores = self.v(scores).squeeze(-1)

        alpha = torch.softmax(scores / self.temperature, dim=1)
        alpha = self.dropout(alpha)

        return (x * alpha.unsqueeze(-1)).sum(dim=1)

# -------------------------------
# MODEL
# -------------------------------
class EEGModel(nn.Module):
    def __init__(self, C, F, num_classes=3):
        super().__init__()

        self.spectral = SpectralWeighting(C, F)

        self.gat_heads = 4
        self.gat_out = 16

        self.gat = MultiHeadGAT(F, self.gat_out, num_heads=self.gat_heads)

        self.lstm = nn.LSTM(
            input_size=self.gat_out * self.gat_heads,
            hidden_size=32,
            batch_first=True,
            bidirectional=True
        )

        self.temporal_attn = TemporalAttention(64)
        self.channel_pool = ChannelAttentionPooling(64)

        self.norm = nn.LayerNorm(64)

        # 🔥 CLASSIFICATION HEAD
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.spectral(x)

        A_list = []
        for i in range(x.shape[0]):
            A_i = compute_graph(x[i])
            A_i = sparsify_graph(A_i)
            A_list.append(A_i)

        A = torch.stack(A_list).to(x.device)

        gat_out = []
        attn_list = []

        for t in range(x.shape[1]):
            out, attn = self.gat(x[:, t], A)
            gat_out.append(out)
            attn_list.append(attn)

        attn_stack = torch.stack(attn_list, dim=1)
        self.last_attention = attn_stack.mean(dim=1).mean(dim=-1)

        x = torch.stack(gat_out, dim=1)

        B, T, C, Fp = x.shape
        x = x.view(B * C, T, Fp)

        x, _ = self.lstm(x)
        x = self.temporal_attn(x)

        x = x.view(B, C, 64)
        x = self.channel_pool(x)
        x = self.norm(x)

        logits = self.fc(x)
        return logits

# -------------------------------
# TRAINING
# -------------------------------
def train_model(model, X_train, y_train, epochs=100, batch_size=16):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss {total_loss/total:.4f} | Acc {correct/total:.3f}")

# -------------------------------
# EVALUATION
# -------------------------------
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        return (logits.argmax(1) == y).float().mean().item()

# -------------------------------
# ATTENTION GRAPH UTILS
# -------------------------------
def sparsify_attention(A, k=3):
    C = A.shape[0]
    A_sparse = torch.zeros_like(A)

    for i in range(C):
        vals, idx = torch.topk(A[i], k)
        for j in idx:
            if i != j:
                A_sparse[i, j] = A[i, j]
                A_sparse[j, i] = A[i, j]

    return A_sparse

def plot_graph(A, title="Graph"):
    A = A.detach().cpu().numpy()
    C = A.shape[0]

    G = nx.Graph()
    for i in range(C):
        G.add_node(i)

    for i in range(C):
        for j in range(i + 1, C):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=A[i, j])

    labels = {i: CHANNEL_LABELS[i] for i in range(C)}
    weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]

    plt.figure(figsize=(6, 6))
    nx.draw(G, FIXED_POS, labels=labels, node_size=700,
            node_color="skyblue", width=weights)

    plt.title(title)
    plt.show()

def build_attention_graph(model, X, y, cls):
    graphs = []

    model.eval()
    with torch.no_grad():
        for i in range(X.shape[0]):
            if y[i].item() == cls:
                _ = model(X[i].unsqueeze(0))
                A = model.last_attention.squeeze(0)
                graphs.append(A)

    if len(graphs) == 0:
        return None

    A = torch.stack(graphs).mean(0)
    A = A / (A.max() + 1e-8)
    A = sparsify_attention(A)

    return A


def evaluate_class_metrics(model, X, y, target_class):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)

    # Convert to CPU numpy
    preds = preds.cpu()
    y = y.cpu()

    # True/False masks
    tp = ((preds == target_class) & (y == target_class)).sum().item()
    fp = ((preds == target_class) & (y != target_class)).sum().item()
    fn = ((preds != target_class) & (y == target_class)).sum().item()

    # Metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)   # ← your “class accuracy”
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


# -------------------------------
# MAIN
# -------------------------------
def main():
    X_train = torch.load("../data/processed/X_train.pt").to(device)
    y_train = torch.load("../data/processed/y_train.pt").to(device)

    X_test  = torch.load("../data/processed/X_test.pt").to(device)
    y_test  = torch.load("../data/processed/y_test.pt").to(device)

    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    model = EEGModel(X_train.shape[2], X_train.shape[3]).to(device)

    train_model(model, X_train, y_train)

    train_acc = evaluate(model, X_train, y_train)
    test_acc  = evaluate(model, X_test, y_test)

    print("\nFINAL RESULTS")
    print(f"Train Acc: {train_acc:.3f}")
    print(f"Test  Acc: {test_acc:.3f}")

    print("\nPlotting FINAL attention graphs...\n")

    CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}

    for cls in range(3):
        A_cls = build_attention_graph(model, X_train, y_train, cls)

        if A_cls is not None:
            plot_graph(A_cls, f"{CLASS_NAMES[cls]} Attention Graph")

    print("\nCLASS-WISE METRICS\n")

    CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}

    for cls in range(3):
        p, r, f1 = evaluate_class_metrics(model, X_test, y_test, cls)

        print(f"{CLASS_NAMES[cls]}:")
        print(f"  Precision: {p:.3f}")
        print(f"  Recall (Class Accuracy): {r:.3f}")
        print(f"  F1 Score: {f1:.3f}\n")

if __name__ == "__main__":
    main()