import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, stft
from pathlib import Path
import mne
from mne.preprocessing import ICA

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

N_CHANNELS = len(SELECTED_CHANNELS)
ICA_N_COMPONENTS = N_CHANNELS - 1


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
    b, a = butter(4, [LOWCUT / nyq, HIGHCUT / nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)


# ===============================
# ICA
# ===============================
def apply_ica_mne(signal):
    if not hasattr(np, "in1d"):
        np.in1d = np.isin

    ch_names = [ch.split('-')[0] for ch in SELECTED_CHANNELS]
    info = mne.create_info(ch_names=ch_names, sfreq=FS, ch_types="eeg")

    raw = mne.io.RawArray(signal.T, info, verbose=False)

    def highpass(sig, cutoff=1.0):
        nyq = 0.5 * FS
        b, a = butter(4, cutoff / nyq, btype='high')
        return filtfilt(b, a, sig, axis=0)

    signal_hp = highpass(signal)
    raw_for_ica = mne.io.RawArray(signal_hp.T, info, verbose=False)

    ica = ICA(
        n_components=ICA_N_COMPONENTS,
        random_state=42,
        max_iter=500
    )
    ica.fit(raw_for_ica, verbose=False)

    sources = ica.get_sources(raw_for_ica).get_data()

    variances = np.var(sources, axis=1)
    kurtosis  = np.mean(
        (sources - sources.mean(axis=1, keepdims=True)) ** 4, axis=1
    )

    var_norm  = variances / (variances.max() + 1e-8)
    kurt_norm = kurtosis  / (kurtosis.max()  + 1e-8)
    score     = var_norm  + kurt_norm

    threshold      = np.percentile(score, 90)
    bad_components = np.where(score > threshold)[0]

    max_remove     = max(1, int(0.2 * ICA_N_COMPONENTS))
    bad_components = bad_components[:max_remove]
    ica.exclude    = list(bad_components)

    cleaned_raw = raw.copy()
    ica.apply(cleaned_raw)

    return cleaned_raw.get_data().T


# ===============================
# SEGMENTS
# ===============================
def extract_segments(trial):
    start = 3 * FS
    return {
        "low":  trial[start           : start + 5 * FS],
        "mid":  trial[start + 5 * FS  : start + 10 * FS],
        "high": trial[start + 10 * FS : start + 15 * FS],
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
    freq_mask  = f <= 40
    stft_array = stft_array[:, freq_mask, :]

    return stft_array


# ===============================
# NORMALIZE
# ===============================
def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-6)


# ===============================
# PROCESS ONE FILE
# ===============================
def process_file(filepath):
    data     = load_data(filepath)
    filtered = bandpass_filter(data)
    filtered = apply_ica_mne(filtered)

    total_samples = filtered.shape[0]

    X_list        = []
    y_list        = []
    trial_id_list = []
    trial_counter = 0

    for i in range(0, total_samples, TRIAL_LENGTH):
        trial = filtered[i : i + TRIAL_LENGTH]

        if len(trial) < TRIAL_LENGTH:
            continue

        trial_counter += 1
        segments = extract_segments(trial)

        for label, seg in segments.items():
            stft_data = compute_stft(seg)

            x = torch.tensor(stft_data, dtype=torch.float32)
            x = x.unsqueeze(0).permute(0, 3, 1, 2)  # (1, 38, 16, 21)
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

    perm          = torch.randperm(len(unique_trials))
    unique_trials = unique_trials[perm]

    split_idx    = int(train_ratio * len(unique_trials))
    train_trials = unique_trials[:split_idx]
    test_trials  = unique_trials[split_idx:]

    train_mask = torch.isin(trial_ids, train_trials)
    test_mask  = torch.isin(trial_ids, test_trials)

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    RAW_DIR  = Path("../data/raw").resolve()
    SAVE_DIR = Path("../data/processed").resolve()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    all_X      = []
    all_y      = []
    all_trials = []

    trial_offset = 0

    for file in sorted(RAW_DIR.glob("*.xlsx")):
        print(f"Processing {file.name}")

        X_list, y_list, trial_list = process_file(file)

        if not X_list:
            print(f"  ⚠ No valid trials in {file.name}")
            continue

        all_X.extend(X_list)
        all_y.extend(y_list)

        max_local_id = max(trial_list)
        all_trials.extend([t + trial_offset for t in trial_list])
        trial_offset += max_local_id

        print(f"  Trials: {max_local_id}   Samples: {len(X_list)}")

    if not all_X:
        raise RuntimeError("No data found.")

    X         = torch.cat(all_X, dim=0)
    y         = torch.tensor(all_y)
    trial_ids = torch.tensor(all_trials)

    print("\nFULL DATASET")
    print("X shape :", X.shape)
    print("y shape :", y.shape)

    # ---- Split ----
    X_train, X_test, y_train, y_test = trial_wise_split(X, y, trial_ids)

    print("\nFINAL SPLIT")
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    # ---- Save ----
    torch.save(X_train, SAVE_DIR / "X_train.pt")
    torch.save(y_train, SAVE_DIR / "y_train.pt")
    torch.save(X_test,  SAVE_DIR / "X_test.pt")
    torch.save(y_test,  SAVE_DIR / "y_test.pt")

    print("\n✔ Saved clean dataset (NO augmentation)")