import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, stft
from pathlib import Path
import random

# ===============================
# PARAMETERS
# ===============================
FS = 500
LOWCUT = 0.5
HIGHCUT = 40
WINDOW = 128
OVERLAP = 64
NFFT = 256

CHUNK_SIZE = None
LABEL_MAP = {"low": 0, "mid": 1, "high": 2}

TRIAL_DURATION = 18
TRIAL_LENGTH = TRIAL_DURATION * FS

SELECTED_CHANNELS = [
    "Fp1-A1","Fp2-A2",
    "F3-A1","F4-A2",
    "C3-A1","C4-A2",
    "P3-A1","P4-A2",
    "O1-A1","O2-A2",
    "F7-A1","F8-A2",
    "T3-A1","T4-A2",
    "T5-A1","T6-A2"
]

# ===============================
# FILTER
# ===============================
def bandpass_filter(signal):
    nyq = 0.5 * FS
    low = LOWCUT / nyq
    high = HIGHCUT / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)

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
# SEGMENT EXTRACTION
# ===============================
def extract_segments(data):
    start = 3 * FS

    low = data[start : start + 5*FS]
    mid = data[start + 5*FS : start + 10*FS]
    high = data[start + 10*FS : start + 15*FS]

    return {"low": low, "mid": mid, "high": high}

# ===============================
# NORMALIZATION
# ===============================
def normalize_tensor(x):
    return (x - x.mean()) / (x.std() + 1e-6)

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
# PROCESS FILE (TRIAL-WISE)
# ===============================
def process_file(filepath):
    print(f"\nProcessing: {filepath}")

    data = load_data(filepath)
    filtered = bandpass_filter(data)

    trials = []
    total_samples = data.shape[0]

    trial_count = 0

    for i in range(0, total_samples, TRIAL_LENGTH):
        trial = filtered[i:i + TRIAL_LENGTH]

        if len(trial) < TRIAL_LENGTH:
            continue

        trial_count += 1
        segments = extract_segments(trial)

        trial_data = []
        trial_labels = []

        for label, seg in segments.items():
            stft_data = compute_stft(seg)

            x = torch.tensor(stft_data, dtype=torch.float32)
            x = x.unsqueeze(0).permute(0, 3, 1, 2)  # (1, T, C, F)
            x = normalize_tensor(x)

            trial_data.append(x)
            trial_labels.append(LABEL_MAP[label])

        trials.append((trial_data, trial_labels))

    print(f"Trials extracted: {trial_count}")
    return trials

# ===============================
# FLATTEN
# ===============================
def flatten_trials(trials):
    data, labels = [], []
    for d, l in trials:
        data.extend(d)
        labels.extend(l)
    return torch.cat(data, dim=0), torch.tensor(labels)

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    RAW_DIR = Path("../data/raw").resolve()
    SAVE_DIR = Path("../data/processed").resolve()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    all_trials = []

    for file in RAW_DIR.glob("*.xlsx"):
        print(f"\nProcessing file: {file.name}")
        trials = process_file(file)
        all_trials.extend(trials)

    print(f"\nTotal trials: {len(all_trials)}")

    # 🔥 SHUFFLE TRIALS
    random.shuffle(all_trials)

    # 🔥 SPLIT
    split = int(0.8 * len(all_trials))
    train_trials = all_trials[:split]
    test_trials  = all_trials[split:]

    # 🔥 FLATTEN AFTER SPLIT
    X_train, y_train = flatten_trials(train_trials)
    X_test, y_test   = flatten_trials(test_trials)

    print("\nFINAL DATASET")
    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    # SAVE
    torch.save(X_train, SAVE_DIR / "X_train.pt")
    torch.save(y_train, SAVE_DIR / "y_train.pt")
    torch.save(X_test, SAVE_DIR / "X_test.pt")
    torch.save(y_test, SAVE_DIR / "y_test.pt")

    print("\nSaved TRAIN/TEST datasets correctly!")