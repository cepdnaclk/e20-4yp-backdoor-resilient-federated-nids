import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1️⃣ Single-Stage Baseline Network
# ============================================================
class Net(nn.Module):
    def __init__(self, input_dim=71, num_classes=10):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        return self.fc4(x)


# ============================================================
# 2️⃣ Wider Single-Stage Network (Better Capacity)
# ============================================================
class WiderNet(nn.Module):
    def __init__(self, input_dim=71, num_classes=10):
        super(WiderNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        self.fc4 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        return self.fc4(x)


# ============================================================
# 3️⃣ Two-Stage Network (Used Separately)
# ============================================================
class TwoStageNet(nn.Module):
    """
    Stage 1: Normal vs Attack (Binary)
    Stage 2: Attack Type (9 classes)
    """

    def __init__(self, input_dim=79):
        super(TwoStageNet, self).__init__()

        # ---------- Stage 1 ----------
        self.stage1_fc1 = nn.Linear(input_dim, 128)
        self.stage1_fc2 = nn.Linear(128, 64)
        self.stage1_fc3 = nn.Linear(64, 32)
        self.stage1_out = nn.Linear(32, 2)

        # ---------- Stage 2 ----------
        self.stage2_fc1 = nn.Linear(input_dim, 128)
        self.stage2_fc2 = nn.Linear(128, 64)
        self.stage2_fc3 = nn.Linear(64, 32)
        self.stage2_out = nn.Linear(32, 9)

        self.dropout = nn.Dropout(0.2)

    def forward_stage1(self, x):
        x = F.relu(self.stage1_fc1(x))
        x = self.dropout(x)
        x = F.relu(self.stage1_fc2(x))
        x = self.dropout(x)
        x = F.relu(self.stage1_fc3(x))
        return self.stage1_out(x)

    def forward_stage2(self, x):
        x = F.relu(self.stage2_fc1(x))
        x = self.dropout(x)
        x = F.relu(self.stage2_fc2(x))
        x = self.dropout(x)
        x = F.relu(self.stage2_fc3(x))
        return self.stage2_out(x)

    def forward(self, x):
        return self.forward_stage1(x), self.forward_stage2(x)
