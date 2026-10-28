"""All figures for Lab 6."""
import numpy as np
import matplotlib.pyplot as plt

from core import QUALITY_THRESHOLDS, REQUIRED_SR, SLOPE_RANGE, psd_db, quantize

DPI = 130


def _finish(fig, axes, path):
    for ax in np.atleast_1d(axes):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)


def plot_fragment(ax, y, sr, t0, dur, **kw):
    """Plot y on [t0, t0 + dur] with time in ms relative to t0."""
    i0 = int(round(t0 * sr))
    i1 = min(i0 + int(dur * sr) + 1, len(y))
    t = (np.arange(i0, i1) / sr - t0) * 1000
    ax.plot(t, y[i0:i1], **kw)


def plot_original(y, sr, t0, frag, path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    axes[0].plot(np.arange(len(y)) / sr, y, lw=0.5)
    axes[0].axvspan(t0, t0 + frag, color="red", alpha=0.3, label="zoomed fragment")
    axes[0].set(title=f"Original signal (sr = {sr} Hz)", xlabel="time, s", ylabel="amplitude")
    axes[0].legend(loc="upper right")
    plot_fragment(axes[1], y, sr, t0, frag, marker=".", ms=4, lw=1)
    axes[1].set(title=f"{frag * 1000:g} ms fragment - individual samples are visible",
                xlabel="time, ms", ylabel="amplitude")
    _finish(fig, axes, path)


def plot_resampled_time(y, sr, signals, t0, frag, path):
    rates = [fs for fs in REQUIRED_SR if fs in signals]
    fig, axes = plt.subplots(len(rates), 1, figsize=(11, 2.3 * len(rates)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, fs in zip(axes, rates):
        plot_fragment(ax, y, sr, t0, frag, color="lightgray", lw=2, label="original")
        plot_fragment(ax, signals[fs][0], fs, t0, frag, marker="o", ms=3, lw=1, label=f"{fs} Hz")
        ax.set_ylabel(f"{fs} Hz")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title("Same fragment at different sampling rates (with anti-aliasing filter)")
    axes[-1].set_xlabel("time, ms")
    _finish(fig, axes, path)


def plot_aliasing_spectrum(y, sr, y_filt, y_naive, fs_low, path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    f, p = psd_db(y, sr)
    axes[0].plot(f, p, lw=0.8)
    axes[0].axvline(fs_low / 2, color="red", ls="--",
                    label=f"Nyquist for {fs_low} Hz = {fs_low / 2:g} Hz")
    axes[0].set(title=f"Spectrum of the original (sr = {sr} Hz)",
                xlabel="frequency, Hz", ylabel="PSD, dB")
    axes[0].legend()

    f, p = psd_db(y_filt, fs_low)
    axes[1].plot(f, p, lw=1, label="with anti-aliasing filter")
    f, p = psd_db(y_naive, fs_low)
    axes[1].plot(f, p, lw=1, alpha=0.8, label="without filter (aliasing)")
    axes[1].set(title=f"Spectrum after sampling at {fs_low} Hz: components above "
                      f"{fs_low / 2:g} Hz fold back into the band",
                xlabel="frequency, Hz", ylabel="PSD, dB")
    axes[1].legend()
    _finish(fig, axes, path)


def plot_quantization_time(y, sr, demo_bits, t0, frag, path):
    fig, axes = plt.subplots(len(demo_bits), 1, figsize=(11, 2.3 * len(demo_bits)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, b in zip(axes, demo_bits):
        plot_fragment(ax, y, sr, t0, frag, color="lightgray", lw=2, label="original")
        plot_fragment(ax, quantize(y, b), sr, t0, frag, lw=1, drawstyle="steps-post",
                      label=f"{b}-bit quantization")
        ax.set_ylabel(f"{b} bit ({2 ** b} lvls)")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title("Effect of quantization at different bit depths")
    axes[-1].set_xlabel("time, ms")
    _finish(fig, axes, path)


def plot_snr_vs_fs(rows, sr, f99, path):
    x = np.array([r["fs"] for r in rows]) / 1000
    s_f = np.array([r["snr_filt"] for r in rows], dtype=float)
    s_n = np.array([r["snr_naive"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    fin = np.isfinite(s_f)
    ax.plot(x[fin], s_f[fin], "o-", color="C0", label="with anti-aliasing filter")
    ax.plot(x, s_n, "s--", color="C1", label="without filter (aliasing)")

    # fs == original rate -> identity, SNR = inf. Draw it at the top instead of dropping it.
    inf_mask = np.isinf(s_f)
    if inf_mask.any():
        finite_vals = np.concatenate([s_f[fin], s_n[np.isfinite(s_n)]])
        top = finite_vals.max() + 10 if finite_vals.size else 100
        ax.plot(x[inf_mask], np.full(inf_mask.sum(), top), "^", color="C0", ms=10,
                label=f"fs = original rate ({sr} Hz): no change, SNR = inf")
        for xi in x[inf_mask]:
            ax.annotate("inf", (xi, top), textcoords="offset points", xytext=(0, 8), ha="center")

    ax.axvline(2 * f99 / 1000, color="gray", ls=":",
               label=f"2 * f_max(99%) = {2 * f99 / 1000:.1f} kHz")
    ax.set(title="SNR = f(sampling rate)", xlabel="sampling rate, kHz", ylabel="SNR, dB")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x], rotation=45)
    ax.legend(fontsize=8)
    _finish(fig, ax, path)


def plot_snr_vs_bits(rows, p_signal, path):
    bits = [r["bits"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(bits, [r["snr"] for r in rows], "o-", label="measured SNR")
    ax.plot(bits, [r["theory_sine"] for r in rows], "--",
            label="theory 6.02*b + 1.76 (full-scale sine)")
    ax.plot(bits, [r["theory_signal"] for r in rows], ":", color="C3",
            label=f"theory for this signal's power (P = {p_signal:.3f})")
    lo, hi = SLOPE_RANGE
    ax.axvspan(lo, hi, color="C2", alpha=0.07, label=f"slope fit range {lo}-{hi} bits")
    for thr, name in QUALITY_THRESHOLDS:
        ax.axhline(thr, color="gray", ls=":", lw=1)
        ax.text(bits[0], thr + 1, f"{thr} dB {name}", fontsize=8, color="gray")
    ax.set(title="SNR = f(bit depth)", xlabel="bit depth, bits", ylabel="SNR, dB")
    ax.set_xticks(bits)
    ax.legend(fontsize=8)
    _finish(fig, ax, path)
