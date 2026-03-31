import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, stft
from pathlib import Path

# ===============================
# PARAMETERS
# ===============================
FS = 500
LOWCUT = 0.5
HIGHCUT = 40

WINDOW = 128
OVERLAP = 64
NFFT = 256

TRIAL_DURATION = 18
TRIAL_LENGTH = TRIAL_DURATION * FS

LABEL_MAP = {"low": 0, "mid": 1, "high": 2}

SELECTED_CHANNELS = [
    "Fp1-A1","Fp2-A2","F3-A1","F4-A2",
    "C3-A1","C4-A2","P3-A1","P4-A2",
    "O1-A1","O2-A2","F7-A1","F8-A2",
    "T3-A1","T4-A2","T5-A1","T6-A2"
]

# ===============================
# LOAD DATA
# ===============================
def load_data(filepath):
    df = pd.read_excel(filepath)

    missing = [ch for ch in SELECTED_CHANNELS if ch not in df.columns]
    if missing:
        raise ValueError(f"Missing channels: {missing}")

    return df[SELECTED_CHANNELS].values


# ===============================
# FILTER
# ===============================
def bandpass_filter(signal):
    nyq = 0.5 * FS
    b, a = butter(4, [LOWCUT/nyq, HIGHCUT/nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)


# ===============================
# SEGMENTS
# ===============================
def extract_segments(trial):
    start = 3 * FS
    return {
        "low":  trial[start : start + 5*FS],
        "mid":  trial[start + 5*FS : start + 10*FS],
        "high": trial[start + 10*FS : start + 15*FS]
    }


# ===============================
# STFT
# ===============================
def compute_stft(segment):
    C = segment.shape[1]
    stft_list = []

    for ch in range(C):
        f, t, Zxx = stft(
            segment[:, ch],
            fs=FS,
            nperseg=WINDOW,
            noverlap=OVERLAP,
            nfft=NFFT
        )
        stft_list.append(np.abs(Zxx))

    stft_array = np.stack(stft_list, axis=0)

    freq_mask = f <= 40
    stft_array = stft_array[:, freq_mask, :]

    return stft_array


# ===============================
# NORMALIZE
# ===============================
def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-6)


# ===============================
# PROCESS FILE (NO AUGMENTATION)
# ===============================
def process_file(filepath):

    data = load_data(filepath)
    filtered = bandpass_filter(data)

    total_samples = filtered.shape[0]

    X_list = []
    y_list = []
    trial_id_list = []

    trial_counter = 0

    for i in range(0, total_samples, TRIAL_LENGTH):
        trial = filtered[i:i + TRIAL_LENGTH]

        if len(trial) < TRIAL_LENGTH:
            continue

        trial_counter += 1
        segments = extract_segments(trial)

        for label, seg in segments.items():

            stft_data = compute_stft(seg)

            x = torch.tensor(stft_data, dtype=torch.float32)
            x = x.unsqueeze(0).permute(0, 3, 1, 2)  # (1, T, C, F)
            x = normalize(x)

            X_list.append(x)
            y_list.append(LABEL_MAP[label])
            trial_id_list.append(trial_counter)

    return X_list, y_list, trial_id_list


# ===============================
# TRIAL-WISE SPLIT
# ===============================
def trial_wise_split(X, y, trial_ids, train_ratio=0.8):

    unique_trials = torch.unique(trial_ids)

    perm = torch.randperm(len(unique_trials))
    unique_trials = unique_trials[perm]

    split_idx = int(train_ratio * len(unique_trials))

    train_trials = unique_trials[:split_idx]
    test_trials  = unique_trials[split_idx:]

    train_mask = torch.isin(trial_ids, train_trials)
    test_mask  = torch.isin(trial_ids, test_trials)

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


# ===============================
# SAFE AUGMENTATION (TRAIN ONLY)
# ===============================
def augment_tensor(x):
    augmented = []

    # original
    augmented.append(x)

    # small noise
    augmented.append(x + torch.randn_like(x) * 0.02)

    # slightly stronger noise
    augmented.append(x + torch.randn_like(x) * 0.04)

    # mild scaling
    augmented.append(x * (0.9 + 0.2 * torch.rand(1)))

    # stronger scaling
    augmented.append(x * (0.8 + 0.4 * torch.rand(1)))

    return augmented


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    RAW_DIR = Path("../data/raw").resolve()
    SAVE_DIR = Path("../data/processed").resolve()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    all_X = []
    all_y = []
    all_trials = []

    for file in RAW_DIR.glob("*.xlsx"):
        print(f"Processing {file.name}")

        X_list, y_list, trial_list = process_file(file)

        all_X.extend(X_list)
        all_y.extend(y_list)
        all_trials.extend(trial_list)

    X = torch.cat(all_X, dim=0)
    y = torch.tensor(all_y)
    trial_ids = torch.tensor(all_trials)

    print("\nFULL DATASET")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Trials:", len(torch.unique(trial_ids)))

    # ===============================
    # SPLIT
    # ===============================
    X_train, X_test, y_train, y_test = trial_wise_split(X, y, trial_ids)

    print("\nBefore Augmentation")
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    # ===============================
    # 🔥 AUGMENT TRAIN ONLY
    # ===============================
    X_train_aug = []
    y_train_aug = []

    for i in range(len(X_train)):
        augmented = augment_tensor(X_train[i])

        for aug in augmented:
            X_train_aug.append(aug.unsqueeze(0))
            y_train_aug.append(y_train[i])

    X_train = torch.cat(X_train_aug, dim=0)
    y_train = torch.tensor(y_train_aug)

    # Shuffle train
    perm = torch.randperm(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    print("\nAfter Augmentation")
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    # ===============================
    # SAVE
    # ===============================
    torch.save(X_train, SAVE_DIR / "X_train.pt")
    torch.save(y_train, SAVE_DIR / "y_train.pt")
    torch.save(X_test, SAVE_DIR / "X_test.pt")
    torch.save(y_test, SAVE_DIR / "y_test.pt")

    print("\nSaved clean dataset with correct augmentation!")