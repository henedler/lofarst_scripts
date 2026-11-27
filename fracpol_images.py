#!/usr/bin/env python3
"""Fractional circular polarisation (V/I) images from a wsclean -pol IV run.

For a wsclean image name (e.g. img/img-bp) this reads every channel image and the MFS image
(<name>-<chan>-I-image.fits / -V-image.fits and <name>-MFS-I/V-image.fits, or <name>-I/V-image.fits
for a single channel) and writes

    <name>-<chan>-VoverI.fits   V/I for every pixel (NaN only where I = 0 or I/V are not finite)
    <name>-<chan>-VoverI.png    plot in a diverging colour map (default -0.5 ... +0.5; values beyond
                                the scale are shown in the end colours)
    <name>-VoverI-overview.png  all channels and the MFS image on one page

No pixels are blanked on signal-to-noise: where I is faint V/I can take any value.

    fracpol_images.py img/img-bp [--vmax 0.5]
"""
import argparse
import glob
import os
import re

import numpy as np
from astropy.io import fits
import astropy
from astropy.utils.iers import conf
conf.auto_max_age = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap             # noqa: E402

INK = "#3d3d3a"
MASKED = "#c8c7c0"
# diverging: blue (negative) - neutral grey - red (positive)
CMAP = LinearSegmentedColormap.from_list("v_over_i", ["#0f4c99", "#257de0", "#efefef", "#ec4647", "#a3201f"])
CMAP.set_bad(MASKED)


def find_channels(name):
    """[(label, I file, V file)] for all channels (sorted) and the MFS image"""
    out = []
    for f in sorted(glob.glob(f"{glob.escape(name)}-[0-9][0-9][0-9][0-9]-I-image.fits")):
        chan = re.search(r"-(\d{4})-I-image\.fits$", f).group(1)
        out.append((chan, f, f.replace("-I-image.fits", "-V-image.fits")))
    for label, stem in (("MFS", f"{name}-MFS"), ("single", name)):
        i = f"{stem}-I-image.fits"
        if os.path.exists(i) and (label == "MFS" or not out):
            out.append((label, i, f"{stem}-V-image.fits"))
    return [o for o in out if os.path.exists(o[2])]


def frac_v(ifile, vfile):
    with fits.open(ifile) as hi, fits.open(vfile) as hv:
        I = np.squeeze(hi[0].data).astype(float)
        V = np.squeeze(hv[0].data).astype(float)
        header = hi[0].header.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        q = V / I
    q[~np.isfinite(q)] = np.nan                                     # I = 0 (e.g. outside the imaged sky)
    return q, header


def freq_label(header):
    for k in range(1, header.get("NAXIS", 0) + 1):
        if header.get(f"CTYPE{k}", "").upper().startswith("FREQ"):
            return f"{header[f'CRVAL{k}'] / 1e6:.2f} MHz"
    return ""


def extent_deg(header, shape):
    """image extent as offsets from the reference pixel in degrees (SIN direction cosines, east left)"""
    ny, nx = shape
    x0 = (0.5 - header.get("CRPIX1", nx / 2)) * header.get("CDELT1", -1)
    x1 = (nx + 0.5 - header.get("CRPIX1", nx / 2)) * header.get("CDELT1", -1)
    y0 = (0.5 - header.get("CRPIX2", ny / 2)) * header.get("CDELT2", 1)
    y1 = (ny + 0.5 - header.get("CRPIX2", ny / 2)) * header.get("CDELT2", 1)
    return [x0, x1, y0, y1]


def draw(ax, q, vmax, title, header):
    im = ax.imshow(q, origin="lower", cmap=CMAP, vmin=-vmax, vmax=vmax, interpolation="nearest",
                   extent=extent_deg(header, q.shape))
    ax.set_title(title, color=INK, fontsize=9)
    for s in ax.spines.values():
        s.set_edgecolor("#bdbcb5")
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="wsclean -name of the imaging run, e.g. img/img-bp")
    ap.add_argument("--vmax", type=float, default=0.5, help="colour scale -vmax ... +vmax")
    args = ap.parse_args()

    chans = find_channels(args.name)
    if not chans:
        raise SystemExit(f"no wsclean I and V images found for {args.name}")
    plt.rcParams.update({"font.size": 8, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    results = []
    for label, ifile, vfile in chans:
        q, header = frac_v(ifile, vfile)
        stem = f"{args.name}-{label}" if label != "single" else args.name
        hdr = header.copy()
        hdr["BUNIT"] = ""
        hdr["HISTORY"] = f"V/I from {os.path.basename(vfile)} / {os.path.basename(ifile)}"
        out = np.full(header["NAXIS2"] * header["NAXIS1"], np.nan).reshape(header["NAXIS2"], header["NAXIS1"])
        out[:] = q
        fits.writeto(f"{stem}-VoverI.fits", out.reshape((1,) * (header["NAXIS"] - 2) + out.shape).astype(np.float32),
                     hdr, overwrite=True)
        title = f"{os.path.basename(args.name)} {label}  {freq_label(header)}"
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        im = draw(ax, q, args.vmax, title, header)
        ax.set_xlabel("offset in RA from the phase centre (deg, SIN)")
        ax.set_ylabel("offset in Dec (deg, SIN)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="both")
        cb.set_label("V / I", color=INK)
        fig.savefig(f"{stem}-VoverI.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        results.append((label, q, header))

    n = len(results)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow + 0.6), squeeze=False)
    for ax in axs.flat[n:]:
        ax.axis("off")
    for ax, (label, q, header) in zip(axs.flat, results):
        im = draw(ax, q, args.vmax, f"{label}  {freq_label(header)}", header)
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.025, pad=0.02, extend="both")
    cb.set_label("V / I", color=INK)
    fig.suptitle(f"{os.path.basename(args.name)}: fractional circular polarisation V/I", color=INK)
    fig.savefig(f"{args.name}-VoverI-overview.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"{args.name}: V/I FITS and PNGs for {', '.join(r[0] for r in results)}")


if __name__ == "__main__":
    main()
