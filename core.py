"""
Lab 6 core: settings, WAV I/O, signal processing, experiments and table output.
"""
import csv

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import welch

REQUIRED_SR = [8000, 16000, 22050, 44100]
# Extra points that make the SNR(fs) curve smoother (used only if <= original rate)
EXTRA_SR = [4000, 6000, 11025, 32000, 48000]

# Bit depths to study. Depths >= the file's native PCM depth are skipped automatically
# re-quantizing a 16-bit file to 16 bits only shifts every sample by half a step,
# which is not quantization noise and gives a misleading SNR point.
BITS = list(range(1, 17))
DEMO_BITS = [2, 4, 8, 16]            # shown on the time-domain quantization plot
LISTEN_BITS = [2, 4, 6, 8, 12, 16]   # saved as WAV for listening

# Bit range used to fit the "dB per bit" slope.
# At 1-3 bits the uniform-noise model of quantization error does not hold.
SLOPE_RANGE = (4, 14)

# Seconds trimmed from both ends when computing resampling SNR (filter transients)
EDGE_SEC = 0.01

# Reference SNR levels for "acceptable quality"
QUALITY_THRESHOLDS = [
    (40, "~ speech / telephony"),
    (60, "~ acceptable music"),
]

# Test-signal parameters 
TEST_SR, TEST_DURATION, TEST_F0, TEST_SUBTYPE = 44100, 2.0, 440.0, "PCM_24"

# libsndfile subtype -> bits per sample
SUBTYPE_BITS = {
    "PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
    "FLOAT": 32, "DOUBLE": 64, "ULAW": 8, "ALAW": 8,
}


# =============================================================================
# WAV I/O
# =============================================================================
def make_test_signal(path, sr=TEST_SR, dur=TEST_DURATION, f0=TEST_F0, subtype=TEST_SUBTYPE):
    """Band-limited sawtooth: f0 plus harmonics with 1/n amplitudes up to ~0.45*sr.

    A rich spectrum (up to ~20 kHz) is needed so that losses at low sampling rates
    and aliasing are actually visible. A pure tone or a tone with a couple of
    overtones fits into 8 kHz completely and would show no difference.
    """
    t = np.arange(int(sr * dur)) / sr
    y = np.zeros_like(t)
    n = 1
    while n * f0 < 0.45 * sr:
        y += np.sin(2 * np.pi * n * f0 * t) / n
        n += 1
    y *= 0.9 / np.max(np.abs(y))          # peak at 0.9 -> no clipping
    sf.write(str(path), y, sr, subtype=subtype)
    print(f"Generated test WAV: {path} ({n - 1} harmonics, f0 = {f0:.0f} Hz, {subtype})")
    return path


def describe(path):
    """Print and return the WAV parameters required by the assignment."""
    info = sf.info(str(path))
    bits = SUBTYPE_BITS.get(info.subtype)
    print("\n=== WAV file parameters ===")
    print(f"File:              {path}")
    print(f"Sample rate:       {info.samplerate} Hz")
    print(f"Channels:          {info.channels}")
    print(f"Format/bit depth:  {info.subtype} ({bits if bits else 'unknown'} bits per sample)")
    print(f"Duration:          {info.duration:.3f} s")
    print(f"Frames:            {info.frames}")
    return info


def load_mono(path):
    """Load as float64 in [-1, 1]; multichannel files are mixed down to mono."""
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    if data.shape[1] > 1:
        print("Multichannel file: analysis uses a mono mix of all channels.")
    return data.mean(axis=1), sr


def save_wav(path, y, sr):
    """Save as 32-bit float so that saving does not add its own quantization."""
    sf.write(str(path), y, sr, subtype="FLOAT")


# =============================================================================
# Signal processing
# =============================================================================
def snr_db(ref, test, margin=0):
    """SNR = 10*log10( mean(x^2) / mean((x - x_hat)^2) ).

    `margin` samples are ignored at both ends (filter transients).
    Returns +inf when the signals are identical.
    """
    n = min(len(ref), len(test))
    ref, test = ref[margin:n - margin], test[margin:n - margin]
    p_noise = np.mean((ref - test) ** 2)
    if p_noise == 0:
        return np.inf
    return 10 * np.log10(np.mean(ref ** 2) / p_noise)


def signal_power(y):
    """Mean power mean(x^2)."""
    return float(np.mean(y ** 2))


def resample(y, sr_from, sr_to):
    """Proper resampling with an anti-aliasing low-pass filter (soxr, zero delay)."""
    if sr_from == sr_to:
        return y.copy()
    return librosa.resample(y, orig_sr=sr_from, target_sr=sr_to, res_type="soxr_hq")


def naive_sample(y, sr, fs):
    """Sampling WITHOUT an anti-aliasing filter: just read the signal at times k/fs.

    Components above fs/2 are not removed, so they fold back (aliasing).
    Linear interpolation is used when sr/fs is not an integer.
    """
    t_old = np.arange(len(y)) / sr
    t_new = np.arange(int(len(y) * fs / sr)) / fs
    return np.interp(t_new, t_old, y)


def quantize(x, bits):
    """Uniform mid-rise quantizer with 2^bits levels on [-1, 1].

    Levels are placed at the centres of 2^bits equal intervals, so the
    error for in-range values is within +-step/2.
    """
    levels = 2 ** bits
    step = 2.0 / levels
    idx = np.clip(np.floor((np.clip(x, -1, 1) + 1) / step), 0, levels - 1)
    return -1 + (idx + 0.5) * step


def theoretical_snr(bits, p_signal=0.5):
    """Theoretical SNR of a uniform quantizer on [-1, 1].

    step = 2 / 2^b, noise power = step^2 / 12, so
    SNR = 10*log10(p_signal / (step^2/12)) = 6.02*b + 10*log10(3*p_signal).
    For a full-scale sine (p_signal = 0.5) this is the classic 6.02*b + 1.76 dB.
    """
    step = 2.0 / 2 ** np.asarray(bits, dtype=float)
    return 10 * np.log10(p_signal / (step ** 2 / 12))


def bandwidth(y, sr, frac=0.99):
    """Frequency below which `frac` of the signal energy lies (effective f_max)."""
    p = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    c = np.cumsum(p) / np.sum(p)
    return f[min(np.searchsorted(c, frac), len(f) - 1)]


def psd_db(y, sr):
    """Power spectral density (Welch), in dB."""
    f, p = welch(y, fs=sr, nperseg=min(4096, len(y)))
    return f, 10 * np.log10(p + 1e-20)


# =============================================================================
# Experiments
# =============================================================================
def run_sampling(y, sr, margin, wav_dir):
    """Resample to every rate of the grid and measure SNR.

    SNR is measured by bringing the signal back to the original rate and comparing
    it with the original (signals of different lengths cannot be compared directly).
    Two variants:
      * filtered - proper resampling with an anti-aliasing filter;
      * naive    - plain sampling without a filter (only for fs < sr).
    """
    grid = sorted(set(REQUIRED_SR) | {f for f in EXTRA_SR if f <= sr})
    rows, signals = [], {}
    for fs in grid:
        y_f = resample(y, sr, fs)
        snr_f = snr_db(y, resample(y_f, fs, sr), margin)

        y_n, snr_n = None, np.nan
        if fs < sr:
            y_n = naive_sample(y, sr, fs)
            snr_n = snr_db(y, resample(y_n, fs, sr), margin)

        if fs in REQUIRED_SR:
            signals[fs] = (y_f, y_n)
            save_wav(wav_dir / f"resampled_{fs}Hz.wav", y_f, fs)
            if y_n is not None:
                save_wav(wav_dir / f"naive_{fs}Hz_aliasing.wav", y_n, fs)

        rows.append({"fs": fs, "samples": len(y_f), "nyquist": fs / 2,
                     "snr_filt": snr_f, "snr_naive": snr_n})
    return rows, signals


def usable_bits(bits, info):
    """Drop bit depths >= the native PCM depth of the file. Returns (kept, skipped)."""
    native = SUBTYPE_BITS.get(info.subtype)
    if not (info.subtype.startswith("PCM") and native):
        return list(bits), []
    return [b for b in bits if b < native], [b for b in bits if b >= native]


def run_quantization(y, sr, bits_grid, wav_dir):
    """Quantize to every bit depth and compare SNR with theory."""
    p = signal_power(y)
    rows = []
    for b in bits_grid:
        yq = quantize(y, b)
        rows.append({
            "bits": b,
            "levels": 2 ** b,
            "snr": snr_db(y, yq),
            "theory_sine": float(theoretical_snr(b)),       # 6.02b + 1.76
            "theory_signal": float(theoretical_snr(b, p)),  # same formula, real signal power
        })
        if b in LISTEN_BITS:
            save_wav(wav_dir / f"quantized_{b}bit.wav", yq, sr)
    return rows


# Tables
def fmt(v, digits=2):
    """Format a number for tables: inf -> 'inf', NaN/None -> '-'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if np.isinf(v):
        return "inf"
    return f"{v:.{digits}f}"


def print_table(title, headers, rows):
    print(f"\n=== {title} ===")
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = " | ".join(str(h).rjust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(c).rjust(w) for c, w in zip(r, widths)))


def save_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def sampling_table(rows):
    headers = ["fs, Hz", "samples", "Nyquist fs/2, Hz", "SNR filtered, dB", "SNR no filter, dB"]
    body = [[r["fs"], r["samples"], f"{r['nyquist']:g}", fmt(r["snr_filt"]), fmt(r["snr_naive"])]
            for r in rows]
    return headers, body


def quantization_table(rows):
    headers = ["bits", "levels", "SNR measured, dB", "theory sine 6.02b+1.76, dB",
               "theory this signal, dB"]
    body = [[r["bits"], r["levels"], fmt(r["snr"]), fmt(r["theory_sine"]), fmt(r["theory_signal"])]
            for r in rows]
    return headers, body
