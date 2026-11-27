#!/usr/bin/env python3
"""Verify the resonance correction with a second full-Jones solve on the resonance-corrected data.

Compares, per dipole, the off-diagonal kappa = H_XY / mean(H_XX, H_YY) of the Hermitian part H of
  before     the full-Jones solutions on the bandpass-corrected data (amplitude000/phase000 in cal-resonance.h5)
  model      the resonance model (--model soltabs in cal-resonance.h5, default amplitudeResonance/phaseResonance)
  expected   before - model (to first order what the verification solve should find)
  verify     the full-Jones solutions on the resonance-corrected data (cal-verify.h5)
If the resonance model describes the data, verify ~ expected, its rms in the resonance band drops to the
fit residual, and nothing of the resonance shape is left (projection ~ 0).

    plot_verify.py cal-resonance.h5 cal-verify.h5 [--outdir plots-verify] [--summary verify_summary.txt]

Writes <outdir>/verify_summary.png (medians over dipoles and typical |kappa| against frequency),
<outdir>/verify_dipoles.pdf (one panel per dipole) and the summary table.
"""
import argparse
import os
import sys

import numpy as np
import tables
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from plot_resonance import jones, hermitian_part, fit_range       # noqa: E402

INK, GRID, GREY, BLUE, ORANGE, SHADE, RED = ("#3d3d3a", "#e6e5e0", "#8a8980", "#2a78d6", "#eb6834",
                                             "#f3f2ee", "#b23b1e")
BANDS = ((30, 45), (45, 57), (57, 67.5))


def kappa(J):
    H = hermitian_part(J.copy())
    with np.errstate(invalid="ignore", divide="ignore"):
        return H[..., 0, 1] / (0.5 * (H[..., 0, 0].real + H[..., 1, 1].real))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("resonance_h5", help="h5parm with the full-Jones solutions and the resonance soltabs")
    ap.add_argument("verify_h5", help="h5parm of the full-Jones solve on the resonance-corrected data")
    ap.add_argument("--solset", default="sol000")
    ap.add_argument("--model", default="amplitudeResonance,phaseResonance",
                    help="resonance model amplitude,phase soltabs in the first h5parm")
    ap.add_argument("--outdir", default="plots-verify")
    ap.add_argument("--summary", default="verify_summary.txt")
    ap.add_argument("--fmin", type=float, help="lower edge of the resonance fit range (Hz) "
                                                "(default: from the history of the resonance soltab)")
    args = ap.parse_args()

    with tables.open_file(args.resonance_h5, "r") as h5:
        J0, _, freq, ants = jones(h5, args.solset, "amplitude000", "phase000")
        JM, wm, _, _ = jones(h5, args.solset, *args.model.split(","))
        if args.fmin is None:
            args.fmin = fit_range(h5, args.solset, args.model.split(",")[0])[0]
    with tables.open_file(args.verify_h5, "r") as h5:
        JV, _, freqv, antsv = jones(h5, args.solset, "amplitude000", "phase000")
    if len(freqv) != len(freq) or not np.allclose(freqv, freq) or list(antsv) != list(ants):
        raise SystemExit("the two h5parms have different frequencies or antennas")
    nu = freq / 1e6
    k0, kv, km = kappa(J0), kappa(JV), JM[..., 0, 1]
    kexp = k0 - km
    fitted = np.any(np.abs(km) > 0, axis=0)
    infit = nu >= args.fmin / 1e6
    ok = fitted & (np.isfinite(k0[infit]).mean(axis=0) > 0.8) & (np.isfinite(kv[infit]).mean(axis=0) > 0.8)
    g = np.where(ok)[0]
    if g.size == 0:
        raise SystemExit("no dipole with a resonance fit and unflagged solutions in both h5parms")

    # ---- outlier channels (e.g. one bad snapshot) are left out of the statistics: channels where the
    # median over dipoles of kappa (before or verify) jumps away from its running median
    def outliers(k):
        """channels where the median over dipoles jumps away from its running median, or where the
        scatter over dipoles is far above the typical scatter (single bad channels/snapshots)"""
        med_k = np.nanmedian(k[:, g].real, axis=1) + 1j * np.nanmedian(k[:, g].imag, axis=1)
        c = np.abs(med_k)
        run = np.array([np.nanmedian(c[max(0, i - 4):i + 5]) for i in range(len(c))])
        d = np.abs(c - run)
        common = (d > 5 * 1.4826 * np.nanmedian(d)) & (d > 0.05)
        spread = np.sqrt(np.nanmedian(np.abs(k[:, g] - med_k[:, None]) ** 2, axis=1))
        scatter = spread > 5 * np.nanmedian(spread)
        return common | scatter
    clean = ~(outliers(k0) | outliers(kv))

    # ---- numbers
    rms = lambda a, m: np.sqrt(np.nanmean(np.abs(a[m & clean][:, g]) ** 2, axis=0))
    noise = np.nanmedian(np.sqrt(np.nanmean(np.abs(np.diff(k0[(nu > 45) & (nu < 55)][:, g], axis=0)) ** 2, axis=0) / 2))
    med = lambda a: np.nanmedian(a[:, g].real, axis=1) + 1j * np.nanmedian(a[:, g].imag, axis=1)
    lines = [f"# verification of {args.resonance_h5} ({args.model}) with {args.verify_h5}: {g.size} dipoles",
             "# outlier channels left out of the numbers: " +
             (", ".join(f"{x:.2f}" for x in nu[~clean]) + " MHz" if (~clean).any() else "none"),
             f"# kappa = H_XY / mean(H_XX, H_YY); channel-to-channel noise of kappa (45-55 MHz): {noise:.4f}",
             "# rms over the band per dipole, median over dipoles; 'common' = largest |median over dipoles| in the band",
             f"# {'band_MHz':>9s} {'before':>8s} {'expected':>9s} {'verify':>8s} {'verify-exp':>10s} {'common_before':>13s} {'common_verify':>13s}"]
    for lo, hi in BANDS:
        m = (nu >= lo) & (nu < hi)
        if not m.any():
            continue
        lines.append(f"  {lo:4.0f}-{hi:<4.0f} {np.median(rms(k0, m)):8.4f} {np.median(rms(kexp, m)):9.4f} "
                     f"{np.median(rms(kv, m)):8.4f} {np.median(rms(kv - kexp, m)):10.4f} "
                     f"{np.nanmax(np.abs(med(k0)[m & clean])):13.4f} {np.nanmax(np.abs(med(kv)[m & clean])):13.4f}")
    mres = nu >= args.fmin / 1e6
    def projection(i):
        m = mres & clean & np.isfinite(kv[:, i]) & np.isfinite(km[:, i])
        return np.real(np.vdot(km[m, i], kv[m, i]) / np.vdot(km[m, i], km[m, i]))
    proj = np.array([projection(i) for i in g])
    lines.append(f"# projection of verify onto the fitted resonance above {args.fmin / 1e6:g} MHz "
                 f"(1 = nothing removed, 0 = all removed): median {np.median(proj):.3f}, "
                 f"p10 {np.percentile(proj, 10):.3f}, p90 {np.percentile(proj, 90):.3f}")
    with open(args.summary, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- summary figure
    os.makedirs(args.outdir, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#bdbcb5", "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    fig, ax = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    for a, f, lab in ((ax[0], np.real, "Re κ"), (ax[1], np.imag, "Im κ")):
        a.plot(nu, f(med(k0)), color=GREY, lw=1.4, label="before (full-Jones on bandpass-corrected data)")
        a.plot(nu, f(med(kexp)), color=BLUE, lw=1.4, ls="--", label="expected: before − resonance model")
        a.plot(nu, f(med(kv)), color=ORANGE, lw=1.6, label="verify (full-Jones on resonance-corrected data)")
        a.set_ylabel(lab + " (median over dipoles)")
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")
    typ = lambda a: np.sqrt(np.nanmedian(np.abs(a[:, g]) ** 2, axis=1))
    ax[2].plot(nu, typ(k0), color=GREY, lw=1.4, label="before")
    ax[2].plot(nu, typ(kexp), color=BLUE, lw=1.4, ls="--", label="expected")
    ax[2].plot(nu, typ(kv), color=ORANGE, lw=1.6, label="verify")
    ax[2].axhline(noise, color="#bdbcb5", lw=1, ls=":")
    ax[2].text(nu.min() + 0.3, noise * 1.3, f"noise ({noise:.4f})", color=GREY, fontsize=8)
    ax[2].set_ylabel("|κ| of a typical dipole")
    ax[2].set_xlabel("Frequency (MHz)")
    ax[2].legend(frameon=False, fontsize=8)
    for a in ax:
        if args.fmin / 1e6 > nu.min():
            a.axvspan(nu.min(), args.fmin / 1e6, color=SHADE, zorder=-3, lw=0)
        a.grid(color=GRID, lw=0.6)
        a.set_axisbelow(True)
    ax[0].set_title(f"Resonance verification ({args.model.split(',')[0]}), {g.size} dipoles (shaded: outside the resonance fit range)", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "verify_summary.png"), dpi=100)
    plt.close(fig)

    # ---- per dipole
    ylim = np.nanpercentile(np.abs(np.concatenate([k0[:, g].real.ravel(), k0[:, g].imag.ravel()])), 99.5) * 1.15
    with PdfPages(os.path.join(args.outdir, "verify_dipoles.pdf")) as pdf:
        for p0 in range(0, len(ants), 16):
            fig, axs = plt.subplots(4, 4, figsize=(14, 10), sharex=True, sharey=True)
            for a, i in zip(axs.flat, range(p0, p0 + 16)):
                if i >= len(ants):
                    a.axis("off")
                    continue
                if args.fmin / 1e6 > nu.min():
                    a.axvspan(nu.min(), args.fmin / 1e6, color=SHADE, zorder=-3, lw=0)
                a.grid(color=GRID, lw=0.6)
                a.set_axisbelow(True)
                a.plot(nu, k0[:, i].real, color=GREY, lw=0.8, alpha=0.7, label="before Re")
                a.plot(nu, k0[:, i].imag, color=GREY, lw=0.8, ls="--", alpha=0.7, label="before Im")
                a.plot(nu, kv[:, i].real, ".", ms=2.5, color=BLUE, label="verify Re")
                a.plot(nu, kv[:, i].imag, ".", ms=2.5, color=ORANGE, label="verify Im")
                if i in g:
                    m = nu >= args.fmin / 1e6
                    a.set_title(f"{ants[i]}  rms {np.sqrt(np.nanmean(np.abs(k0[m, i]) ** 2)):.3f} → "
                                f"{np.sqrt(np.nanmean(np.abs(kv[m, i]) ** 2)):.3f}", color=INK, fontsize=8)
                else:
                    a.set_title(f"{ants[i]}  no resonance fit / flagged", color=RED, fontsize=8)
            axs.flat[0].set_ylim(-ylim, ylim)
            axs.flat[0].legend(frameon=False, fontsize=7, loc="lower left")
            for a in axs[-1]:
                a.set_xlabel("Frequency (MHz)")
            for a in axs[:, 0]:
                a.set_ylabel("κ")
            fig.suptitle("κ before (lines) and after the resonance correction (dots); title: rms in the fit range",
                         color=INK)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {os.path.join(args.outdir, 'verify_summary.png')}, {os.path.join(args.outdir, 'verify_dipoles.pdf')}")


if __name__ == "__main__":
    main()
