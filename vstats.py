#!/usr/bin/env python3
"""Stokes V flux statistics of several wsclean -pol IV image sets, with one common mask.

    vstats.py img/img-bp img/img-fj img/img-res [--ref img/img-bp] [--ms MS] [--min-elev 10] [--snr 5]

For every channel (and the MFS image) one mask is built from the reference image set and applied
to all image sets, so that the numbers are directly comparable:
  - elevation >= --min-elev (pixel positions converted to alt/az for the middle of the observation,
    station position from the MS; without --ms the phase centre is taken as the zenith),
  - significant Stokes I in the reference: I_ref > --snr x sigma_I, with sigma_I the robust rms of the
    reference I residual image (wsclean -residual.fits) above the elevation limit.
In the masked area, per image set and channel (fluxes in Jy, i.e. pixel sums divided by the beam area):
  S_I        integrated Stokes I
  V_pos      sum of the positive V pixels
  V_neg      sum of the negative V pixels
  V_net      V_pos + V_neg
  V_abs      V_pos - V_neg  (= sum |V|)
  V_noise    expected sum |V| from noise alone: N_pix x sigma_V x sqrt(2/pi) / beam, sigma_V = robust rms of
             the V image outside the I mask (above the elevation limit)
  V_excess   V_abs - V_noise, the |V| flux above the noise
  Vnet/I, Vexc/I   V_net / S_I and V_excess / S_I
  slope      least-squares slope of V against I over the masked pixels (I -> V leakage), with offset
Results are printed and written to --out (default: vstats.txt next to the first image set).
"""
import argparse
import glob
import os
import re

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy
from astropy.utils.iers import conf
conf.auto_max_age = None


def channels(name):
    labels = [re.search(r"-(\d{4})-I-image\.fits$", f).group(1)
              for f in sorted(glob.glob(f"{glob.escape(name)}-[0-9][0-9][0-9][0-9]-I-image.fits"))]
    if os.path.exists(f"{name}-MFS-I-image.fits"):
        labels.append("MFS")
    if not labels and os.path.exists(f"{name}-I-image.fits"):
        labels.append("")
    return labels


def fname(name, label, pol, kind="image"):
    return f"{name}-{label}-{pol}-{kind}.fits" if label else f"{name}-{pol}-{kind}.fits"


def read(fn):
    with fits.open(fn) as h:
        return np.squeeze(h[0].data).astype(float), h[0].header


def robust_rms(a):
    a = a[np.isfinite(a)]
    return 1.4826 * np.median(np.abs(a - np.median(a))) if a.size else np.nan


def pix_per_beam(h):
    return np.pi * h["BMAJ"] * h["BMIN"] / (4 * np.log(2)) / abs(h["CDELT1"] * h["CDELT2"])


def elevation_map(header, ms=None):
    """elevation (deg) of every pixel; NaN outside the projection"""
    wcs = WCS(header).celestial
    ny, nx = header["NAXIS2"], header["NAXIS1"]
    yy, xx = np.mgrid[:ny, :nx]
    ra, dec = wcs.all_pix2world(xx, yy, 0)
    ra, dec = np.radians(ra), np.radians(dec)
    if ms:
        from astropy.coordinates import EarthLocation, AltAz, SkyCoord
        from astropy.time import Time
        from astropy.utils import iers
        import astropy.units as u
        from casacore import tables as ct
        iers.conf.auto_download = False
        iers.conf.auto_max_age = None
        if hasattr(iers.conf, "iers_degraded_accuracy"):
            iers.conf.iers_degraded_accuracy = "warn"
        pos = ct.table(ms + "/ANTENNA", ack=False).getcol("POSITION").mean(axis=0)
        t = ct.table(ms, ack=False).getcol("TIME")
        loc = EarthLocation.from_geocentric(*pos, unit="m")
        when = Time(0.5 * (t.min() + t.max()) / 86400.0, format="mjd", scale="utc")
        zen = SkyCoord(AltAz(az=0 * u.deg, alt=90 * u.deg, obstime=when, location=loc)).icrs
        ra0, dec0 = zen.ra.rad, zen.dec.rad
    else:
        ra0, dec0 = np.radians(header["CRVAL1"]), np.radians(header["CRVAL2"])
    cosz = np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0)
    return np.degrees(np.arcsin(np.clip(cosz, -1, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="wsclean -name of the image sets, e.g. img/img-bp img/img-fj")
    ap.add_argument("--ref", help="image set that defines the Stokes I mask (default: the first one)")
    ap.add_argument("--ms", help="measurement set, for the zenith (default: phase centre = zenith)")
    ap.add_argument("--min-elev", type=float, default=10.0, help="elevation limit in degrees")
    ap.add_argument("--snr", type=float, default=5.0, help="I_ref > snr x sigma_I(residual)")
    ap.add_argument("--out", help="output table (default: vstats.txt next to the first image set)")
    args = ap.parse_args()

    names = [n for n in args.names if channels(n)]
    missing = [n for n in args.names if n not in names]
    if missing:
        print(f"no images found for {missing}, skipped")
    ref = args.ref or names[0]
    labels = channels(ref)
    out = args.out or os.path.join(os.path.dirname(names[0]) or ".", "vstats.txt")

    cols = ["image", "chan", "freq_MHz", "N_pix", "S_I", "V_pos", "V_neg", "V_net", "V_abs", "V_noise",
            "V_excess", "Vnet/I", "Vexc/I", "slope"]
    rows = []
    elev = None
    for label in labels:
        Iref, href = read(fname(ref, label, "I"))
        if elev is None or elev.shape != Iref.shape:
            elev = elevation_map(href, args.ms)
        sky = np.isfinite(elev) & (elev >= args.min_elev)
        res_file = fname(ref, label, "I", "residual")
        res = read(res_file)[0] if os.path.exists(res_file) else Iref
        sigma_i = robust_rms(res[sky])
        mask = sky & np.isfinite(Iref) & (Iref > args.snr * sigma_i)
        freq = next((href[f"CRVAL{k}"] / 1e6 for k in range(1, href["NAXIS"] + 1)
                     if href.get(f"CTYPE{k}", "").upper().startswith("FREQ")), np.nan)
        for name in names:
            try:
                I, h = read(fname(name, label, "I"))
                V, _ = read(fname(name, label, "V"))
            except FileNotFoundError:
                continue
            if I.shape != mask.shape:
                print(f"{name} {label}: other image size than the reference, skipped")
                continue
            ppb = pix_per_beam(h)
            m = mask & np.isfinite(I) & np.isfinite(V)
            sigma_v = robust_rms(V[sky & ~mask])
            v = V[m]
            s_i = I[m].sum() / ppb
            v_pos = v[v > 0].sum() / ppb
            v_neg = v[v < 0].sum() / ppb
            v_abs = v_pos - v_neg
            v_noise = m.sum() * sigma_v * np.sqrt(2 / np.pi) / ppb
            slope = np.polyfit(I[m], v, 1)[0] if m.sum() > 2 else np.nan
            rows.append([os.path.basename(name), label or "-", freq, int(m.sum()), s_i, v_pos, v_neg, v_pos + v_neg,
                         v_abs, v_noise, v_abs - v_noise, (v_pos + v_neg) / s_i, (v_abs - v_noise) / s_i, slope])

    fmt = lambda r: (f"{r[0]:<12s} {r[1]:>5s} {r[2]:8.3f} {r[3]:7d} " + " ".join(f"{x:10.1f}" for x in r[4:11])
                     + " " + " ".join(f"{x:8.4f}" for x in r[11:]))
    header = (f"# Stokes V statistics; mask from {os.path.basename(ref)}: elevation >= {args.min_elev:g} deg"
              f"{' (zenith from ' + os.path.basename(args.ms) + ')' if args.ms else ' (phase centre = zenith)'}"
              f", I_ref > {args.snr:g} sigma_I(residual); fluxes in Jy\n"
              f"# {cols[0]:<10s} {cols[1]:>5s} {cols[2]:>8s} {cols[3]:>7s} " + " ".join(f"{c:>10s}" for c in cols[4:11])
              + " " + " ".join(f"{c:>8s}" for c in cols[11:]))
    with open(out, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(fmt(r) + "\n")
    print(header)
    for r in rows:
        if r[1] in ("MFS", "-"):
            print(fmt(r))
    print(f"all channels: {out}")


if __name__ == "__main__":
    main()
