"""
BrainDepth GNN-BiLSTM — EEG Preprocessing Pipeline (Excel / 500 Hz)
=====================================================================
Protocol (500 Hz):
  fixation cross  : 3s × 500 = 1500 samples  → discard
  low depth       : 5s × 500 = 2500 samples  → class 0 (Near)
  mid depth       : 5s × 500 = 2500 samples  → class 1 (Mid)
  far depth       : 5s × 500 = 2500 samples  → class 2 (Far)
  one cycle total :            9000 samples
  85001 rows      → 9 complete cycles → 9 trials × 3 classes per sheet

Pipeline: Load → Bandpass (3–13 Hz, elliptic order-10) → Segment → STFT
Output  : single HDF5  X[total_trials, 16, F', T_frames]  y[total_trials]
"""

import gc
from pathlib import Path
import numpy as np
import h5py
import openpyxl
import pandas as pd
from scipy.signal import ellip, sosfiltfilt, stft
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
WORKBOOK_1  = Path("../data/raw/trial1.xlsx").resolve()
WORKBOOK_2  = Path("../data/raw/trial2.xlsx").resolve()
OUTPUT_H5   = Path("../data/processed/braindepth.h5").resolve()

FS           = 500
N_CHANNELS   = 16
N_CLASSES    = 3
CLASS_NAMES  = ["Near", "Mid", "Far"]

CH_O1 = 8
CH_O2 = 9

FIXATION_SAMPLES  = 3 * FS
STIMULUS_SAMPLES  = 5 * FS
CYCLE_SAMPLES     = FIXATION_SAMPLES + N_CLASSES * STIMULUS_SAMPLES  # 9000

PLOT_SAMPLES = 2000

BP_LOW    = 3.0
BP_HIGH   = 13.0
BP_ORDER  = 10

STFT_WINDOW   = 1024
STFT_OVERLAP  = 896
STFT_HOP      = STFT_WINDOW - STFT_OVERLAP


# ─────────────────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────────────────
def _build_sos() -> np.ndarray:
    return ellip(
        N=BP_ORDER, rp=0.5, rs=60,
        Wn=[BP_LOW, BP_HIGH], btype="bandpass", fs=FS, output="sos",
    )

SOS = _build_sos()


def bandpass_filter(eeg: np.ndarray) -> np.ndarray:
    out = np.empty_like(eeg)
    for ch in range(eeg.shape[1]):
        out[:, ch] = sosfiltfilt(SOS, eeg[:, ch])
    return out


# ─────────────────────────────────────────────────────────
# STFT
# ─────────────────────────────────────────────────────────
_FREQ_MASK = None

def compute_stft(window: np.ndarray) -> np.ndarray:
    global _FREQ_MASK
    stacks = []
    for ch in range(window.shape[0]):
        freqs, _, Zxx = stft(
            window[ch], fs=FS, window="hann",
            nperseg=STFT_WINDOW, noverlap=STFT_OVERLAP,
            boundary="zeros", padded=True,
        )
        if _FREQ_MASK is None:
            _FREQ_MASK = (freqs >= BP_LOW) & (freqs <= BP_HIGH)
            print(f"    STFT: F'={_FREQ_MASK.sum()} bins "
                  f"({freqs[_FREQ_MASK][0]:.2f}–{freqs[_FREQ_MASK][-1]:.2f} Hz)")
        stacks.append(np.abs(Zxx[_FREQ_MASK, :]).astype(np.float32))
    return np.stack(stacks, axis=0)


# ─────────────────────────────────────────────────────────
# LOAD ONE SHEET
# ─────────────────────────────────────────────────────────
def load_sheet(workbook_path: Path, sheet_name: str):
    df = pd.read_excel(
        workbook_path, sheet_name=sheet_name,
        header=0, engine="openpyxl",
    )
    if df.empty or df.shape[0] < CYCLE_SAMPLES:
        print(f"    Skipping empty/short sheet '{sheet_name}'")
        return None
    eeg = df.iloc[:, :N_CHANNELS].values.astype(np.float32)
    if eeg.shape[1] != N_CHANNELS:
        raise ValueError(
            f"Expected {N_CHANNELS} channels, got {eeg.shape[1]} "
            f"in {workbook_path.name}:{sheet_name}"
        )
    print(f"    Loaded: {eeg.shape[0]} samples x {eeg.shape[1]} channels")
    return eeg


# ─────────────────────────────────────────────────────────
# SEGMENT + STFT
# ─────────────────────────────────────────────────────────
def segment_and_stft(eeg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_cycles  = eeg.shape[0] // CYCLE_SAMPLES
    remainder = eeg.shape[0] % CYCLE_SAMPLES

    if n_cycles == 0:
        print("    No complete cycles found.")
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    print(f"    {n_cycles} cycles → {n_cycles * N_CLASSES} trials "
          f"({remainder} remainder samples discarded)")

    X_list, y_list = [], []
    for cycle in range(n_cycles):
        base = cycle * CYCLE_SAMPLES + FIXATION_SAMPLES
        for cls_idx in range(N_CLASSES):
            start  = base + cls_idx * STIMULUS_SAMPLES
            end    = start + STIMULUS_SAMPLES
            window = eeg[start:end, :].T
            X_list.append(compute_stft(window))
            y_list.append(cls_idx)

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int32)


# ─────────────────────────────────────────────────────────
# PROCESS ONE SHEET
# ─────────────────────────────────────────────────────────
def process_sheet(workbook_path: Path, sheet_name: str):
    eeg = load_sheet(workbook_path, sheet_name)
    if eeg is None:
        return None, None
    eeg = bandpass_filter(eeg)
    X, y = segment_and_stft(eeg)
    del eeg
    gc.collect()
    return X, y


# ─────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────
def run_pipeline():
    OUTPUT_H5.parent.mkdir(parents=True, exist_ok=True)

    all_sheets = []
    subj_id = 1
    for wb_path in [WORKBOOK_1, WORKBOOK_2]:
        if not wb_path.exists():
            print(f"Workbook not found: {wb_path}, skipping.")
            continue
        wb = openpyxl.load_workbook(wb_path, read_only=True, data_only=True)
        for sname in wb.sheetnames:
            all_sheets.append((wb_path, sname, subj_id))
            subj_id += 1
        wb.close()

    print(f"\nBrainDepth Preprocessing Pipeline")
    print(f"  Subjects : {len(all_sheets)}")
    print(f"  Output   : {OUTPUT_H5}\n")

    with h5py.File(OUTPUT_H5, "w") as hf:

        cfg = hf.create_group("config")
        cfg.attrs["fs"]               = FS
        cfg.attrs["n_channels"]       = N_CHANNELS
        cfg.attrs["bp_low"]           = BP_LOW
        cfg.attrs["bp_high"]          = BP_HIGH
        cfg.attrs["stft_window"]      = STFT_WINDOW
        cfg.attrs["stft_hop"]         = STFT_HOP
        cfg.attrs["stimulus_samples"] = STIMULUS_SAMPLES
        cfg.attrs["class_names"]      = CLASS_NAMES

        ds_X = ds_y = ds_sid = None

        for wb_path, sheet_name, sid in tqdm(all_sheets, desc="Subjects", unit="sheet"):
            print(f"\nSubject {sid:02d} | {wb_path.name} → '{sheet_name}'")

            try:
                X, y = process_sheet(wb_path, sheet_name)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            if X is None or X.ndim < 4 or X.shape[0] == 0:
                print("  No trials extracted, skipping.")
                continue

            n_trials = X.shape[0]

            if ds_X is None:
                _, n_ch, F_prime, T_frames = X.shape
                chunk = (1, n_ch, F_prime, T_frames)
                ds_X = hf.create_dataset(
                    "X",
                    shape=(n_trials, n_ch, F_prime, T_frames),
                    maxshape=(None, n_ch, F_prime, T_frames),
                    dtype="float32", chunks=chunk,
                    compression="gzip", compression_opts=4,
                )
                ds_y = hf.create_dataset(
                    "y", shape=(n_trials,), maxshape=(None,),
                    dtype="int32", chunks=(min(n_trials, 512),),
                )
                ds_sid = hf.create_dataset(
                    "subject_id", shape=(n_trials,), maxshape=(None,),
                    dtype="int32", chunks=(min(n_trials, 512),),
                )
                print(f"  HDF5 created: X(n, {n_ch}, {F_prime}, {T_frames})")
                write_start = 0
            else:
                write_start = ds_X.shape[0]
                ds_X.resize(write_start + n_trials, axis=0)
                ds_y.resize(write_start + n_trials, axis=0)
                ds_sid.resize(write_start + n_trials, axis=0)

            ds_X[write_start : write_start + n_trials]   = X
            ds_y[write_start : write_start + n_trials]   = y
            ds_sid[write_start : write_start + n_trials] = sid
            hf.flush()

            counts = dict(zip(*np.unique(y, return_counts=True)))
            print(f"  Written: {n_trials} trials | {counts}")

            del X, y
            gc.collect()

    print("\nPipeline complete.")
    _print_summary(OUTPUT_H5)


# ─────────────────────────────────────────────────────────
# SUMMARY + INSPECT
# ─────────────────────────────────────────────────────────
def _print_summary(h5_path: Path):
    with h5py.File(h5_path, "r") as hf:
        if "X" not in hf:
            print("No data was written to HDF5 — check errors above.")
            return
        X   = hf["X"]
        y   = hf["y"][:]
        sid = hf["subject_id"][:]
        print("─" * 50)
        print(f"Final shape  : X{X.shape}  dtype={X.dtype}")
        print(f"Subjects     : {np.unique(sid).size}")
        for lbl, name in enumerate(CLASS_NAMES):
            print(f"  Class {name:4s} ({lbl}): {(y == lbl).sum()} trials")
        print(f"Uncompressed : {X.size * 4 / 1e6:.1f} MB")
        print("─" * 50)


def inspect_h5(h5_path: Path = OUTPUT_H5):
    with h5py.File(h5_path, "r") as hf:
        for k in hf.keys():
            if k == "config":
                print(f"config: {dict(hf['config'].attrs)}")
            else:
                d = hf[k]
                print(f"{k}: shape={d.shape}  dtype={d.dtype}  chunks={d.chunks}")


if __name__ == "__main__":
    run_pipeline()