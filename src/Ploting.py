"""
plot_preprocess.py
==================
6 separate figures:
  Raw EEG         — O1-A1
  Raw EEG         — O2-A2
  Bandpass 3-13Hz — O1-A1
  Bandpass 3-13Hz — O2-A2
  STFT Mean Power — O1-A1
  STFT Mean Power — O2-A2

Each figure: Near (blue), Mid (orange), Far (green)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import ellip, sosfiltfilt, stft as scipy_stft

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
WORKBOOK    = Path("../data/raw/trial2.xlsx").resolve()
SHEET       = "Sheet1"

FS               = 500
N_CHANNELS       = 16
FIXATION_SAMPLES = 3 * FS
STIMULUS_SAMPLES = 5 * FS
PLOT_SAMPLES     = 2000

BP_LOW   = 3.0
BP_HIGH  = 13.0
BP_ORDER = 10

STFT_WINDOW  = 1024
STFT_OVERLAP = 896

CH_O1 = 8
CH_O2 = 9

CLASS_NAMES  = ["Near", "Mid", "Far"]
CLASS_COLORS = ["royalblue", "darkorange", "seagreen"]


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def build_sos():
    return ellip(N=BP_ORDER, rp=0.5, rs=60,
                 Wn=[BP_LOW, BP_HIGH], btype="bandpass",
                 fs=FS, output="sos")

SOS = build_sos()


def bandpass(eeg):
    out = np.empty_like(eeg)
    for ch in range(eeg.shape[1]):
        out[:, ch] = sosfiltfilt(SOS, eeg[:, ch])
    return out


def stft_mean_power(signal_1d):
    freqs, times, Zxx = scipy_stft(
        signal_1d, fs=FS, window="hann",
        nperseg=STFT_WINDOW, noverlap=STFT_OVERLAP,
        boundary="zeros", padded=True,
    )
    mask = (freqs >= BP_LOW) & (freqs <= BP_HIGH)
    return times, np.abs(Zxx[mask, :]).mean(axis=0)


# ─────────────────────────────────────────────────────────
# LOAD + COMPUTE
# ─────────────────────────────────────────────────────────
print(f"Loading {WORKBOOK.name} -> '{SHEET}' ...")
df  = pd.read_excel(WORKBOOK, sheet_name=SHEET, header=0, engine="openpyxl")
raw = df.iloc[:, :N_CHANNELS].values.astype(np.float32)
print(f"  Shape: {raw.shape}")

print("Bandpass filtering ...")
bp = bandpass(raw.copy())

base = FIXATION_SAMPLES
windows = {}
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    start = base + cls_idx * STIMULUS_SAMPLES
    end   = start + STIMULUS_SAMPLES
    windows[cls_name] = {
        "raw"     : raw[start : start + PLOT_SAMPLES, :],
        "bp"      : bp[start  : start + PLOT_SAMPLES, :],
        "bp_full" : bp[start:end, :],
    }

time_axis = np.arange(PLOT_SAMPLES) / FS


# ─────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────
def make_time_fig(stage_key, ylabel, title):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for cls_name, color in zip(CLASS_NAMES, CLASS_COLORS):
        signal = windows[cls_name][stage_key][:, ch_idx]
        ax.plot(time_axis, signal, color=color,
                linewidth=0.9, alpha=0.85, label=cls_name)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.6, ncol=3)
    plt.tight_layout()
    return fig


def make_stft_fig(title):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for cls_name, color in zip(CLASS_NAMES, CLASS_COLORS):
        times, power = stft_mean_power(windows[cls_name]["bp_full"][:, ch_idx])
        ax.plot(times, power, color=color, linewidth=1.4, label=cls_name)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("STFT Mean Power (3-13 Hz, a.u.)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.6, ncol=3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
# GENERATE — O1-A1
# ─────────────────────────────────────────────────────────
ch_idx = CH_O1
ch_name = "O1-A1"

fig1 = make_time_fig("raw", "Raw EEG (uV)",          f"Raw EEG — {ch_name}")
fig2 = make_time_fig("bp",  "Bandpass 3-13 Hz (uV)", f"Bandpass 3-13 Hz — {ch_name}")
fig3 = make_stft_fig(f"STFT Mean Power — {ch_name}")

# ─────────────────────────────────────────────────────────
# GENERATE — O2-A2
# ─────────────────────────────────────────────────────────
ch_idx = CH_O2
ch_name = "O2-A2"

fig4 = make_time_fig("raw", "Raw EEG (uV)",          f"Raw EEG — {ch_name}")
fig5 = make_time_fig("bp",  "Bandpass 3-13 Hz (uV)", f"Bandpass 3-13 Hz — {ch_name}")
fig6 = make_stft_fig(f"STFT Mean Power — {ch_name}")

plt.show()

# fig1.savefig("raw_O1.png",  dpi=150, bbox_inches="tight")
# fig2.savefig("bp_O1.png",   dpi=150, bbox_inches="tight")
# fig3.savefig("stft_O1.png", dpi=150, bbox_inches="tight")
# fig4.savefig("raw_O2.png",  dpi=150, bbox_inches="tight")
# fig5.savefig("bp_O2.png",   dpi=150, bbox_inches="tight")
# fig6.savefig("stft_O2.png", dpi=150, bbox_inches="tight")