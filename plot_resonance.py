#!/usr/bin/env python3
"""Plot the full-Jones solutions against the fitted resonance model, one panel per dipole.

For every dipole the full-Jones matrix J(nu) (amplitude000 + phase000) is reduced to its Hermitian part
H = (J J^H)^(1/2) (polar decomposition J = H U; only H is constrained by an unpolarised sky model), and
kappa = H_XY / mean(H_XX, H_YY) is plotted (Re, Im; dots) together with the XY element of the model
soltabs written by the losoto RESONANCE operation (lines). Titles give the fit parameters from the
RESONANCE text file if it is given.

    plot_resonance.py cal-resonance.h5 [--fit resonance_fit.txt] [--out resonance_dipoles.pdf]
"""
import argparse
import os
import re

import numpy as np
import tables
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages              # noqa: E402

INK, GRID, BLUE, ORANGE, SHADE, RED = "#3d3d3a", "#e6e5e0", "#2a78d6", "#eb6834", "#f3f2ee", "#b23b1e"


def read_soltab(h5, solset, soltab):
    """Return val, weight with axes (freq, ant, pol) for the first time and direction, plus freq, ant, pol."""
    st = h5.get_node(f"/{solset}/{soltab}")
    axes = st.val.attrs["AXES"]
    axes = (axes.decode() if isinstance(axes, bytes) else axes).split(",")
    def std(a):
        a = np.moveaxis(a, [axes.index(x) for x in ("freq", "ant", "pol")], [-3, -2, -1])
        return a.reshape((-1,) + a.shape[-3:])[0]          # first time/direction
    dec = lambda v: [x.decode() if isinstance(x, bytes) else str(x) for x in v]
    return std(st.val[:]), std(st.weight[:]), st.freq[:], dec(st.ant[:]), dec(st.pol[:])


def jones(h5, solset, amp, ph):
    a, wa, freq, ants, pols = read_soltab(h5, solset, amp)
    p, wp, _, _, _ = read_soltab(h5, solset, ph)
    v = a * np.exp(1j * p)
    v[(wa == 0) | (wp == 0)] = np.nan
    v = v[..., [pols.index(x) for x in ("XX", "XY", "YX", "YY")]]
    return v.reshape(v.shape[:-1] + (2, 2)), (wa > 0).all(axis=-1), freq, ants


def fit_range(h5, solset, soltab, default=(45e6, None)):
    """(fmin, fmax) in Hz of the RESONANCE fit, read from the soltab history (freqRange=a-b MHz)"""
    try:
        attrs = h5.get_node(f"/{solset}/{soltab}").val.attrs
        for name in attrs._f_list("user"):
            if name.startswith("HISTORY"):
                v = attrs[name]
                m = re.search(r"freqRange=([\d.]+)-([\d.]+) MHz", v.decode() if isinstance(v, bytes) else str(v))
                if m:
                    return float(m.group(1)) * 1e6, float(m.group(2)) * 1e6
    except Exception:
        pass
    return default


def hermitian_part(J):
    M = J @ np.conj(np.swapaxes(J, -1, -2))
    bad = ~np.isfinite(M).all(axis=(-1, -2))
    M[bad] = np.eye(2)
    w, V = np.linalg.eigh(M)
    H = (V * np.sqrt(np.clip(w, 0, None))[..., None, :]) @ np.conj(np.swapaxes(V, -1, -2))
    H[bad] = np.nan
    return H


def read_fit(fn):
    """ant -> dict of the RESONANCE text-file columns (first resonance)"""
    if not fn:
        return {}
    cols = None
    out = {}
    for line in open(fn):
        if line.startswith("# ant "):
            cols = line[2:].split()
        elif not line.startswith("#") and cols and line.strip():
            v = line.split()
            out[v[0]] = dict(zip(cols, v))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5parm")
    ap.add_argument("--solset", default="sol000")
    ap.add_argument("--soltabs", default="amplitude000,phase000", help="full-Jones amplitude,phase soltabs")
    ap.add_argument("--model", default="amplitudeResonance,phaseResonance", help="resonance amplitude,phase soltabs")
    ap.add_argument("--fit", help="text file written by RESONANCE (for the panel titles)")
    ap.add_argument("--fmin", type=float, help="lower edge of the fit range (Hz), shaded below "
                                                "(default: from the history of the resonance soltab)")
    ap.add_argument("--out", default="resonance_dipoles.pdf")
    ap.add_argument("--per-page", type=int, default=16)
    args = ap.parse_args()

    with tables.open_file(args.h5parm, "r") as h5:
        J, _, freq, ants = jones(h5, args.solset, *args.soltabs.split(","))
        M, wm, _, _ = jones(h5, args.solset, *args.model.split(","))
        if args.fmin is None:
            args.fmin = fit_range(h5, args.solset, args.model.split(",")[0])[0]
    H = hermitian_part(J)
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = H[..., 0, 1] / (0.5 * (H[..., 0, 0].real + H[..., 1, 1].real))
    model = M[..., 0, 1]
    fit = read_fit(args.fit)
    nu = freq / 1e6

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#bdbcb5", "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    npp = args.per_page
    ncol = int(np.ceil(np.sqrt(npp)))
    nrow = int(np.ceil(npp / ncol))
    ylim = np.nanpercentile(np.abs(np.concatenate([kappa.real.ravel(), kappa.imag.ravel()])), 99.5) * 1.15
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with PdfPages(args.out) as pdf:
        for p0 in range(0, len(ants), npp):
            fig, axs = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.5 * nrow), sharex=True, sharey=True)
            for ax, i in zip(axs.flat, range(p0, p0 + npp)):
                if i >= len(ants):
                    ax.axis("off")
                    continue
                name = ants[i]
                if args.fmin / 1e6 > nu.min():
                    ax.axvspan(nu.min(), args.fmin / 1e6, color=SHADE, zorder=-2, lw=0)
                ax.grid(color=GRID, lw=0.6)
                ax.set_axisbelow(True)
                ax.plot(nu, kappa[:, i].real, ".", ms=2.5, color=BLUE, alpha=0.6)
                ax.plot(nu, kappa[:, i].imag, ".", ms=2.5, color=ORANGE, alpha=0.6)
                r = fit.get(name, {})
                status = r.get("status", "ok" if np.any(np.abs(model[:, i]) > 0) else "no model")
                if status == "ok":
                    ax.plot(nu, model[:, i].real, color=BLUE, lw=1.4, label="Re κ")
                    ax.plot(nu, model[:, i].imag, color=ORANGE, lw=1.4, label="Im κ")
                    title = name
                    if r:
                        title += (f"  ν0 {float(r['nu0_MHz_1']):.2f}  Γ {float(r['Gamma_MHz_1']):.2f} MHz"
                                  f"  A {float(r['A_1']):.2f}  rms {float(r['rms']):.3f}")
                    ax.set_title(title, color=INK, fontsize=8)
                else:
                    ax.set_title(f"{name}  {status}", color=RED, fontsize=8)
                    if not np.isfinite(kappa[:, i]).any():
                        ax.text(0.5, 0.5, "no unflagged solutions", color="#8a8980", ha="center",
                                transform=ax.transAxes)
            axs.flat[0].set_ylim(-ylim, ylim)
            from matplotlib.lines import Line2D
            fig.legend(handles=[Line2D([], [], color=BLUE, marker=".", ls="-", label="Re κ"),
                                Line2D([], [], color=ORANGE, marker=".", ls="-", label="Im κ")],
                       frameon=False, loc="upper right", ncol=2)
            for ax in axs[-1]:
                ax.set_xlabel("Frequency (MHz)")
            for ax in axs[:, 0]:
                ax.set_ylabel("κ")
            fig.suptitle(f"{args.h5parm}: κ = H_XY / mean(H_XX, H_YY) (dots) and RESONANCE model (lines); "
                         f"shaded: outside the fit range", color=INK)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
