import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

LABEL_MAP = {
    "Benign": 0,

    "Generic": 1,
    "Exploits": 2,
    "Fuzzers": 3,
    "DoS": 4,
    "Reconnaissance": 5,
    "Analysis": 6,
    "Backdoor": 7,
    "Shellcode": 8,
    "Worms": 9,
}


DROP_COLUMNS = [
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
]

def load_cic_unsw_nb15(csv_path):
    df = pd.read_csv(csv_path)

    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    df = df.replace([float("inf"), -float("inf")], 0)
    df = df.dropna()

    # Normalize labels
    df["Label"] = df["Label"].astype(str).str.strip()

    # Map labels
    df["Label"] = df["Label"].map(LABEL_MAP)

    # Safety check
    assert df["Label"].isna().sum() == 0, "Unmapped labels detected!"

    X = df.drop(columns=["Label"]).values
    y = df["Label"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        X.shape[1],
    )

