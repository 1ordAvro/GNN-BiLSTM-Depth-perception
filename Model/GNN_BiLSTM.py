import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.weights = nn.Parameter(torch.randn(C, F))

    def forward(self, x):
        w = F.softmax(self.weights, dim=-1)
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
    return torch.stack(A_list).mean(0)

def sparsify_graph(A, k=3):
    C = A.shape[0]
    A_sparse = torch.zeros_like(A)

    for i in range(C):
        vals, idx = torch.topk(A[i], k + 1)
        for j in idx:
            if i != j:
                A_sparse[i, j] = A[i, j]
                A_sparse[j, i] = A[i, j]

    return A_sparse

# -------------------------------
# GAT
# -------------------------------
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.randn(2 * out_features))
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, H, A):
        Wh = self.W(H)
        B, C, _ = Wh.shape

        Wh_i = Wh.unsqueeze(2).repeat(1, 1, C, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, C, 1, 1)

        e = torch.cat([Wh_i, Wh_j], dim=-1)
        e = self.leakyrelu(torch.matmul(e, self.a))

        mask = (A > 0)
        eye = torch.eye(C, device=A.device).bool().unsqueeze(0)
        mask = mask | eye

        e = e.masked_fill(~mask, -1e9)

        alpha = F.softmax(e, dim=-1)
        alpha = F.dropout(alpha, 0.1, training=self.training)

        return F.relu(torch.matmul(alpha, Wh))

# -------------------------------
# CHANNEL ATTENTION
# -------------------------------
class ChannelAttentionPooling(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.attn = nn.Linear(F, 1)

    def forward(self, x):
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=2)
        return (x * weights).sum(dim=2)

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
        scores = self.dropout(scores)

        scores = self.v(scores).squeeze(-1)
        scores = scores / self.temperature

        alpha = torch.softmax(scores, dim=1)
        context = (x * alpha.unsqueeze(-1)).sum(dim=1)

        return context

# -------------------------------
# MODEL
# -------------------------------
class EEGModel(nn.Module):
    def __init__(self, C, F, num_classes=3):
        super().__init__()

        self.spectral = SpectralWeighting(C, F)

        self.gat_out = 32
        self.gat = GATLayer(F, self.gat_out)

        self.pool = ChannelAttentionPooling(self.gat_out)

        self.lstm = nn.LSTM(
            input_size=self.gat_out,
            hidden_size=32,
            batch_first=True,
            bidirectional=True
        )

        self.temporal_attn = TemporalAttention(64)

        self.norm = nn.LayerNorm(64)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.spectral(x)

        A_list = []
        for i in range(x.shape[0]):
            A_i = compute_graph(x[i])
            A_i = sparsify_graph(A_i)
            A_list.append(A_i)

        A = torch.stack(A_list).to(x.device)

        gat_out = []
        for t in range(x.shape[1]):
            gat_out.append(self.gat(x[:, t], A))

        x = torch.stack(gat_out, dim=1)

        x = self.pool(x)

        x, _ = self.lstm(x)

        x = self.temporal_attn(x)

        x = self.norm(x)

        return self.fc(x)

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
            out = model(xb)
            loss = loss_fn(out, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss {total_loss/total:.4f} | Acc {correct/total:.3f}")

# -------------------------------
# EVALUATION
# -------------------------------
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        out = model(X)
        acc = (out.argmax(1) == y).float().mean().item()
    return acc

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

if __name__ == "__main__":
    main()