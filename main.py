"""
Lab 6. Sampling and quantization of an audio signal.

Usage:
    python main.py                       # generate a test WAV and analyse it
    python main.py my_audio.wav          # analyse your own WAV file
    python main.py my.wav --offset 1.0 --frag-ms 8 --no-show

Output -> ./results:
    *.png          time plots, spectra, SNR plots
    snr_*.csv      SNR tables
    wav/*.wav      resampled / quantized signals for listening
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import plots
from core import (BITS, DEMO_BITS, EDGE_SEC, QUALITY_THRESHOLDS, REQUIRED_SR, SLOPE_RANGE,
                  bandwidth, describe, load_mono, make_test_signal, print_table,
                  quantization_table, run_quantization, run_sampling, sampling_table,
                  save_csv, signal_power, theoretical_snr, usable_bits)


def parse_args():
    ap = argparse.ArgumentParser(description="Lab 6: sampling and quantization of audio")
    ap.add_argument("wav", nargs="?", help="path to a WAV file (a test file is generated if omitted)")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--offset", type=float, default=None, help="start of the plotted fragment, s")
    ap.add_argument("--frag-ms", type=float, default=5.0, help="length of the plotted fragment, ms")
    ap.add_argument("--no-show", action="store_true", help="do not open plot windows")
    return ap.parse_args()


def print_conclusions(sampling_rows, quant_rows, sr, f99, p_signal, skipped_bits):
    print("\n=== Conclusions (generated from the measurements) ===")

    below = [r for r in sampling_rows if r["fs"] < sr and np.isfinite(r["snr_filt"])]
    if below:
        lo, hi = below[0], below[-1]
        print(f"1) Sampling rate: with an anti-aliasing filter SNR rises from "
              f"{lo['snr_filt']:.1f} dB at {lo['fs']} Hz to {hi['snr_filt']:.1f} dB at {hi['fs']} Hz. "
              f"A lower fs removes everything above fs/2, so the 'noise' here is lost bandwidth.")

    need = 2 * f99
    ok = [fs for fs in REQUIRED_SR if fs > need]
    print(f"2) Nyquist-Shannon: ~99% of the signal energy lies below f_max ~ {f99:.0f} Hz, "
          f"so fs > {need:.0f} Hz is needed. Required rates that satisfy this: {ok or 'none'}.")

    gaps = [r["snr_filt"] - r["snr_naive"] for r in sampling_rows
            if np.isfinite(r["snr_filt"]) and np.isfinite(r["snr_naive"])]
    if gaps:
        print(f"3) Aliasing: without the filter SNR is lower by {np.mean(gaps):.1f} dB on average "
              f"(up to {max(gaps):.1f} dB). Components above fs/2 fold back into 0..fs/2 "
              f"and can no longer be separated from the useful signal.")

    bits = [r["bits"] for r in quant_rows]
    snr = [r["snr"] for r in quant_rows]
    for thr, name in QUALITY_THRESHOLDS:
        good = [b for b, s in zip(bits, snr) if s >= thr]
        print(f"4) Minimum bit depth for SNR >= {thr} dB ({name}): "
              f"{f'{good[0]} bits' if good else 'not reached'}")

    lo_b, hi_b = SLOPE_RANGE
    mid = [(b, s) for b, s in zip(bits, snr) if lo_b <= b <= hi_b and np.isfinite(s)]
    if len(mid) >= 2:
        xb, ys = zip(*mid)
        slope = np.polyfit(xb, ys, 1)[0]
        print(f"5) Each extra bit adds ~{slope:.2f} dB of SNR "
              f"(fit over {lo_b}-{hi_b} bits; theory: 6.02 dB/bit).")
        offset = np.mean([s - float(theoretical_snr(b)) for b, s in mid])
        expected = 10 * np.log10(p_signal / 0.5)
        print(f"6) Measured SNR differs from the full-scale-sine formula by {offset:+.1f} dB on average; "
              f"expected from signal power: {expected:+.1f} dB (P_signal = {p_signal:.3f} vs 0.5 for "
              f"a full-scale sine). The formula assumes a full-scale sine; lower average power "
              f"means proportionally lower SNR.")

    if skipped_bits:
        print(f"Note: bit depths {skipped_bits} were skipped - quantizing to the file's native depth "
              f"or higher only shifts samples by half a step and is not real quantization noise.")


def main():
    args = parse_args()
    out = Path(args.out)
    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load WAV --------------------------------------------------------
    path = Path(args.wav) if args.wav else make_test_signal(out / "test_signal.wav")

    # ---- 2. File parameters -------------------------------------------------
    info = describe(path)
    y, sr = load_mono(path)
    peak = np.max(np.abs(y))
    if peak == 0:
        raise SystemExit("The file contains only silence - nothing to analyse.")
    p_signal = signal_power(y)
    print(f"Peak level:        {20 * np.log10(peak):.2f} dBFS")
    print(f"RMS level:         {10 * np.log10(p_signal):.2f} dBFS")

    duration = len(y) / sr
    t0 = args.offset if args.offset is not None else min(0.5, duration / 4)
    frag = args.frag_ms / 1000
    margin = int(EDGE_SEC * sr)
    f99 = bandwidth(y, sr)
    print(f"Effective bandwidth (99% energy): f_max ~ {f99:.0f} Hz "
          f"-> Nyquist requires fs > {2 * f99:.0f} Hz")

    plots.plot_original(y, sr, t0, frag, out / "01_original.png")

    # ---- 3. Resampling ------------------------------------------------------
    sampling_rows, signals = run_sampling(y, sr, margin, wav_dir)
    headers, body = sampling_table(sampling_rows)
    print_table("Resampling (SNR after converting back to the original rate)", headers, body)
    save_csv(out / "snr_sampling_rate.csv", headers, body)

    plots.plot_resampled_time(y, sr, signals, t0, frag, out / "02_resampled_time.png")

    fs_low = min(REQUIRED_SR)
    if fs_low < sr:
        y_f, y_n = signals[fs_low]
        plots.plot_aliasing_spectrum(y, sr, y_f, y_n, fs_low, out / "03_aliasing_spectrum.png")

    # ---- 4. Quantization ----------------------------------------------------
    bits_grid, skipped = usable_bits(BITS, info)
    quant_rows = run_quantization(y, sr, bits_grid, wav_dir)
    headers, body = quantization_table(quant_rows)
    print_table("Quantization (SNR vs bit depth)", headers, body)
    save_csv(out / "snr_bit_depth.csv", headers, body)

    demo_bits = [b for b in DEMO_BITS if b in bits_grid]
    plots.plot_quantization_time(y, sr, demo_bits, t0, frag, out / "04_quantization_time.png")

    # ---- 5. Research part: SNR plots ----------------------------------------
    plots.plot_snr_vs_fs(sampling_rows, sr, f99, out / "05_snr_vs_sampling_rate.png")
    plots.plot_snr_vs_bits(quant_rows, p_signal, out / "06_snr_vs_bit_depth.png")

    print_conclusions(sampling_rows, quant_rows, sr, f99, p_signal, skipped)
    print(f"\nAll plots, tables and WAV files saved to: {out.resolve()}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
