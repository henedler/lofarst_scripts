#!/usr/bin/env python3
"""Convert a LOFAR2.0 XST statistics file (HDF5) into a Measurement Set.

Each antenna (dipole pair) becomes an MS antenna, each pair of antennas a
baseline (autocorrelations included). The XST sweep measures every subband once,
at its own 10-s timestamp. Two layouts keep the timing exact:
  --layout single (default): one spectral window with all channels; each
      timestamp holds data only in its own ~7 channels, all others are flagged
      (zero data/weights). Needed for DP3 and EveryBeam, which require exactly
      one spectral window. Larger on disk (compresses well).
  --layout spw: one spectral window per timestamp (~7 channels each). Compact,
      fine for WSClean, but DP3/EveryBeam reject it.

The raw (zenith-referenced, not fringe-stopped) visibilities are phase-rotated
to one fixed J2000 phase centre (default: zenith at the middle of the sweep)
and UVWs are computed for that centre.

Conventions:
    ANTENNA1 = p, ANTENNA2 = q (p <= q)
    UVW      = X_q - X_p projected on the phase-centre frame
    DATA     = conj(XST[2p+a, 2q+b]) after phase rotation (default), which
               images correctly in WSClean. --no-conjugate keeps the raw XST
               sign (gives a point-mirrored sky in WSClean).

EveryBeam (AARTFAAC mode): the MS is written so that EveryBeam treats every
antenna as one LBA dipole, as for AARTFAAC data:
    OBSERVATION::TELESCOPE_NAME       = "AARTFAAC"
    OBSERVATION::AARTFAAC_ANTENNA_TYPE = "LBA"
    ANTENNA keyword AARTFAAC_COORDINATE_AXES = station P, Q, R axes (ITRF) as
        the columns of a 3x3 matrix, from lofarantpos if installed, otherwise
        the local geodetic East, North, Up (within ~0.2 deg for core stations)
    ANTENNA::FLAG_ROW = False for all antennas (bad antennas are flagged in the
        main table instead; EveryBeam asserts on flagged antenna rows)
Use the Hamaker element model (EveryBeam selects 'hamakerlba' for AARTFAAC).

Weights: WEIGHT_SPECTRUM is filled from the autocorrelations with the
radiometer equation. Per real/imaginary component of correlation (a, b) on
baseline (p, q):
    sigma^2 = A_pa * A_qb / (2 * dnu * tau),   weight = 1 / sigma^2
with A the autocorrelation power (XX or YY) of that antenna and channel,
dnu the channel width and tau the integration time. WEIGHT is the channel mean
of WEIGHT_SPECTRUM and SIGMA = 1/sqrt(WEIGHT). Visibilities with a non-positive
autocorrelation get weight 0 and are flagged.

Usage:
    xst2ms.py XST.h5 OUT.ms [--fmin 30] [--fmax 67] [--ra DEG --dec DEG]
              [--time-offset 0] [--no-conjugate]

Requires: numpy, h5py, python-casacore, astropy (for erfa).
"""
import argparse
from datetime import datetime

import h5py
import numpy as np
import erfa
from astropy.time import Time
from casacore import tables as ct

C_LIGHT = 299792458.0


def read_xst(fname):
    with h5py.File(fname, "r") as h5:
        meta = dict(h5["/"].attrs)
        snaps = {}
        for key in h5.keys():
            ds = h5[key]
            t = ds.attrs["timestamp"]
            snaps.setdefault(t, []).append(
                (float(ds.attrs["frequency"]), int(ds.attrs["subband"]),
                 float(ds.attrs["integration_time"]), ds[()]))
    return meta, snaps


def fill_triangle(xst):
    """XST stores a block-lower triangle; fill the rest by Hermitian symmetry."""
    out = xst.astype(np.complex128).copy()
    empty = out == 0
    out[empty] = np.conj(out.T)[empty]
    return out


def c2t(t):
    """Rotation matrix GCRS(=ICRS-aligned) -> ITRS at astropy Time t (no polar motion)."""
    tt, ut1 = t.tt, t.ut1
    return erfa.c2t06a(tt.jd1, tt.jd2, ut1.jd1, ut1.jd2, 0.0, 0.0)


def station_axes(station_field, pos):
    """P, Q, R unit vectors (ITRF) of the antenna field, as rows of a 3x3 array."""
    try:
        from lofarantpos.db import LofarAntennaDatabase
        R = np.asarray(LofarAntennaDatabase().pqr_to_etrs[station_field], dtype=float)
        return R.T, f"PQR axes of {station_field} from lofarantpos"
    except Exception as exc:                                     # noqa: BLE001
        centre = pos.mean(0)
        lon, lat, _ = erfa.gc2gd(1, centre)                       # WGS84 geodetic
        east = np.array([-np.sin(lon), np.cos(lon), 0.0])
        north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
        up = np.cross(east, north)
        return np.stack([east, north, up]), f"local East/North/Up (lofarantpos unavailable: {exc})"


def uvw_basis(ra, dec):
    """Unit vectors (u, v, w) of the phase-centre frame in ICRS axes."""
    sa, ca, sd, cd = np.sin(ra), np.cos(ra), np.sin(dec), np.cos(dec)
    eu = np.array([-sa, ca, 0.0])
    ev = np.array([-sd * ca, -sd * sa, cd])
    ew = np.array([cd * ca, cd * sa, sd])
    return np.stack([eu, ev, ew])  # (3, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xst")
    ap.add_argument("ms")
    ap.add_argument("--fmin", type=float, default=30.0, help="lowest channel frequency to keep (MHz, default 30)")
    ap.add_argument("--fmax", type=float, default=67.0, help="highest channel frequency to keep (MHz, default 67)")
    ap.add_argument("--ra", type=float, help="phase centre RA (deg, J2000); default zenith at mid-sweep")
    ap.add_argument("--dec", type=float, help="phase centre Dec (deg, J2000)")
    ap.add_argument("--time-offset", type=float, default=0.0,
                    help="seconds added to the HDF5 timestamp to get the integration centre")
    ap.add_argument("--layout", choices=("single", "spw"), default="single",
                    help="single spectral window with sparse time coverage (default, works with "
                         "DP3/EveryBeam) or one spectral window per XST timestamp")
    ap.add_argument("--conjugate", action=argparse.BooleanOptionalAction, default=True,
                    help="complex-conjugate the phase-rotated visibilities (default: on; "
                         "--no-conjugate keeps the raw XST sign convention)")
    args = ap.parse_args()

    meta, snaps = read_xst(args.xst)
    pos = np.asarray(meta["antenna_reference_itrf"], dtype=float)       # (nant, 3)
    names = [str(n) for n in meta["antenna_names"]]
    good = np.asarray(meta["antenna_usage_mask"], dtype=bool)[: len(names)]
    nant = len(names)
    p_idx, q_idx = np.triu_indices(nant)                                   # p <= q, autos included
    nbl = len(p_idx)

    # select snapshots fully inside the band, sort by frequency
    sel = []
    for ts, lst in snaps.items():
        lst = sorted(lst)
        f = np.array([x[0] for x in lst])
        keep = (f >= args.fmin * 1e6) & (f <= args.fmax * 1e6)
        if keep.any():
            lst = [x for x, k in zip(lst, keep) if k]
            sel.append((f[keep].min(), ts, lst))
    sel.sort()
    if not sel:
        raise SystemExit("no snapshots inside the requested band")

    times = [Time(datetime.fromisoformat(s[1].split("+")[0]), scale="utc") for s in sel]
    times = [t + args.time_offset / 86400.0 for t in times]
    mjds = np.array([t.mjd for t in times])
    t_mid = Time(0.5 * (mjds.min() + mjds.max()), format="mjd", scale="utc")

    # phase centre: zenith at mid-sweep unless given
    if args.ra is None or args.dec is None:
        R = c2t(t_mid)
        zen_itrs = pos.mean(0) / np.linalg.norm(pos.mean(0))               # geocentric zenith, close enough
        zen_icrs = R.T @ zen_itrs
        ra0 = np.arctan2(zen_icrs[1], zen_icrs[0]) % (2 * np.pi)
        dec0 = np.arcsin(zen_icrs[2])
    else:
        ra0, dec0 = np.radians(args.ra), np.radians(args.dec)
    basis = uvw_basis(ra0, dec0)

    # ---------------------------------------------------------------- tables
    # channel bookkeeping: per block of rows (one XST timestamp) the data
    # description id and where its channels go
    if args.layout == "single":
        chan_f = np.array(sorted(x[0] for s in sel for x in s[2]))
        blocks = []
        for (_, ts, lst), t in sorted(zip(sel, times), key=lambda st: st[1].mjd):
            f = np.array([x[0] for x in lst])
            blocks.append((ts, lst, t, 0, np.searchsorted(chan_f, f), len(chan_f)))
        spws = [chan_f]
        datadesc = ct.maketabdesc([
            ct.makearrcoldesc("DATA", 0j, shape=[len(chan_f), 4], valuetype="complex"),
            ct.makearrcoldesc("WEIGHT_SPECTRUM", 0.0, shape=[len(chan_f), 4], valuetype="float"),
        ])
    else:
        blocks = [(ts, lst, t, i, np.arange(len(lst)), len(lst))
                  for i, ((_, ts, lst), t) in enumerate(zip(sel, times))]
        spws = [np.array([x[0] for x in lst]) for (_, _, lst) in sel]
        datadesc = ct.maketabdesc([
            # snapshots at the band edges keep fewer channels: variable-shape columns
            ct.makearrcoldesc("DATA", 0j, ndim=2, valuetype="complex"),
            ct.makearrcoldesc("WEIGHT_SPECTRUM", 0.0, ndim=2, valuetype="float"),
        ])
    ms = ct.default_ms(args.ms, datadesc)
    ms.close()

    t_ant = ct.table(args.ms + "/ANTENNA", readonly=False, ack=False)
    t_ant.addrows(nant)
    t_ant.putcol("NAME", names)
    t_ant.putcol("STATION", [str(meta["station_name"])] * nant)
    t_ant.putcol("TYPE", ["GROUND-BASED"] * nant)
    t_ant.putcol("MOUNT", ["X-Y"] * nant)
    t_ant.putcol("POSITION", pos)
    t_ant.putcol("OFFSET", np.zeros((nant, 3)))
    t_ant.putcol("DISH_DIAMETER", np.full(nant, 1.0))
    t_ant.putcol("FLAG_ROW", np.zeros(nant, bool))
    station_field = f"{meta['station_name']}{str(meta['antenna_field']).upper()}"
    axes, axes_src = station_axes(station_field, pos)
    # numpy rows P, Q, R become the columns of the casacore matrix (axis order is reversed)
    t_ant.putkeyword("AARTFAAC_COORDINATE_AXES", axes)
    t_ant.close()

    t_pol = ct.table(args.ms + "/POLARIZATION", readonly=False, ack=False)
    t_pol.addrows(1)
    t_pol.putcell("NUM_CORR", 0, 4)
    t_pol.putcell("CORR_TYPE", 0, np.array([9, 10, 11, 12], dtype=np.int32))   # XX XY YX YY
    t_pol.putcell("CORR_PRODUCT", 0, np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32))
    t_pol.putcell("FLAG_ROW", 0, False)
    t_pol.close()

    nspw = len(spws)
    t_spw = ct.table(args.ms + "/SPECTRAL_WINDOW", readonly=False, ack=False)
    t_dd = ct.table(args.ms + "/DATA_DESCRIPTION", readonly=False, ack=False)
    t_spw.addrows(nspw)
    t_dd.addrows(nspw)
    for i, f in enumerate(spws):
        nchan = len(f)
        df = 195312.5
        sb = np.rint(f / df).astype(int)
        t_spw.putcell("NUM_CHAN", i, nchan)
        t_spw.putcell("NAME", i, f"SB{sb[0]:03d}-{sb[-1]:03d}")
        t_spw.putcell("CHAN_FREQ", i, f)
        t_spw.putcell("CHAN_WIDTH", i, np.full(nchan, df))
        t_spw.putcell("EFFECTIVE_BW", i, np.full(nchan, df))
        t_spw.putcell("RESOLUTION", i, np.full(nchan, df))
        t_spw.putcell("REF_FREQUENCY", i, f[0])
        t_spw.putcell("TOTAL_BANDWIDTH", i, df * nchan)
        t_spw.putcell("MEAS_FREQ_REF", i, 5)                       # TOPO
        t_spw.putcell("NET_SIDEBAND", i, 1)
        t_spw.putcell("FLAG_ROW", i, False)
        t_dd.putcell("SPECTRAL_WINDOW_ID", i, i)
        t_dd.putcell("POLARIZATION_ID", i, 0)
        t_dd.putcell("FLAG_ROW", i, False)
    t_spw.close()
    t_dd.close()

    tstart = mjds.min() * 86400.0 - 5.0
    tend = mjds.max() * 86400.0 + 5.0

    t_fld = ct.table(args.ms + "/FIELD", readonly=False, ack=False)
    t_fld.addrows(1)
    t_fld.putcell("NAME", 0, "XST_ZENITH" if args.ra is None else "XST")
    t_fld.putcell("NUM_POLY", 0, 0)
    for col in ("DELAY_DIR", "PHASE_DIR", "REFERENCE_DIR"):
        t_fld.putcell(col, 0, np.array([[ra0, dec0]]))
    t_fld.putcell("TIME", 0, t_mid.mjd * 86400.0)
    t_fld.putcell("FLAG_ROW", 0, False)
    t_fld.close()

    t_feed = ct.table(args.ms + "/FEED", readonly=False, ack=False)
    t_feed.addrows(nant)
    t_feed.putcol("ANTENNA_ID", np.arange(nant))
    t_feed.putcol("FEED_ID", np.zeros(nant, int))
    t_feed.putcol("SPECTRAL_WINDOW_ID", np.full(nant, -1))
    t_feed.putcol("TIME", np.full(nant, t_mid.mjd * 86400.0))
    t_feed.putcol("INTERVAL", np.full(nant, 1e9))
    t_feed.putcol("NUM_RECEPTORS", np.full(nant, 2))
    t_feed.putcol("BEAM_ID", np.full(nant, -1))
    t_feed.putcol("BEAM_OFFSET", np.zeros((nant, 2, 2)))
    t_feed.putcol("POLARIZATION_TYPE", np.array([["X", "Y"]] * nant))
    t_feed.putcol("POL_RESPONSE", np.tile(np.eye(2, dtype=complex), (nant, 1, 1)))
    t_feed.putcol("POSITION", np.zeros((nant, 3)))
    t_feed.putcol("RECEPTOR_ANGLE", np.tile([0.0, np.pi / 2], (nant, 1)))
    t_feed.close()

    t_obs = ct.table(args.ms + "/OBSERVATION", readonly=False, ack=False)
    t_obs.addcols(ct.makescacoldesc("AARTFAAC_ANTENNA_TYPE", "LBA"))
    t_obs.addrows(1)
    t_obs.putcell("TELESCOPE_NAME", 0, "AARTFAAC")
    t_obs.putcell("AARTFAAC_ANTENNA_TYPE", 0, str(meta.get("antenna_type", "LBA")).upper())
    t_obs.putcell("OBSERVER", 0, "XST")
    t_obs.putcell("TIME_RANGE", 0, np.array([tstart, tend]))
    t_obs.putcell("RELEASE_DATE", 0, tend)
    t_obs.putcell("FLAG_ROW", 0, False)
    t_obs.close()

    t_proc = ct.table(args.ms + "/PROCESSOR", readonly=False, ack=False)
    t_proc.addrows(1)
    t_proc.putcell("TYPE", 0, "CORRELATOR")
    t_proc.putcell("FLAG_ROW", 0, False)
    t_proc.close()

    t_state = ct.table(args.ms + "/STATE", readonly=False, ack=False)
    t_state.addrows(1)
    t_state.putcell("FLAG_ROW", 0, False)
    t_state.close()

    # ------------------------------------------------------------- main table
    ms = ct.table(args.ms, readonly=False, ack=False)
    nrow = len(blocks) * nbl
    ms.addrows(nrow)
    bl_itrs = pos[q_idx] - pos[p_idx]                                     # X_q - X_p, (nbl, 3)
    flag_bl = ~(good[p_idx] & good[q_idx])
    a = 2 * p_idx[:, None]
    b = 2 * q_idx[:, None]
    for i, (ts, lst, t, ddid, cidx, nchan_out) in enumerate(blocks):
        R = c2t(t)                                                        # ICRS -> ITRS
        uvw = bl_itrs @ (R @ basis.T)                                     # (nbl, 3)
        freqs = np.array([x[0] for x in lst])
        nchan = len(lst)
        vis = np.empty((nbl, nchan, 4), complex)
        wspec = np.zeros((nbl, nchan, 4))
        dnu_tau = 195312.5 * lst[0][2]
        for c, (_, _, _, xst) in enumerate(lst):
            M = fill_triangle(xst)
            vis[:, c, 0] = M[a[:, 0], b[:, 0]]
            vis[:, c, 1] = M[a[:, 0], b[:, 0] + 1]
            vis[:, c, 2] = M[a[:, 0] + 1, b[:, 0]]
            vis[:, c, 3] = M[a[:, 0] + 1, b[:, 0] + 1]
            # radiometer-equation weights from the autocorrelations
            auto = np.real(np.diag(M))                                    # (2 nant,) X,Y interleaved
            ax, ay = auto[0::2], auto[1::2]
            for k, (ap, aq) in enumerate(((ax, ax), (ax, ay), (ay, ax), (ay, ay))):
                prod = ap[p_idx] * aq[q_idx]
                with np.errstate(divide="ignore"):
                    wspec[:, c, k] = np.where(prod > 0, 2.0 * dnu_tau / prod, 0.0)
        # phase-rotate from local zenith (raw) to the fixed phase centre
        vis *= np.exp(2j * np.pi * uvw[:, 2:3, None] * freqs[None, :, None] / C_LIGHT)
        # conjugate AFTER the rotation, so that the phase centre stays consistent
        if args.conjugate:
            vis = np.conj(vis)

        # place this timestamp's channels in the output channel axis
        vis_out = np.zeros((nbl, nchan_out, 4), complex)
        w_out = np.zeros((nbl, nchan_out, 4))
        vis_out[:, cidx] = vis
        w_out[:, cidx] = wspec
        vis, wspec = vis_out, w_out

        r0 = i * nbl
        ms.putcol("ANTENNA1", p_idx, r0, nbl)
        ms.putcol("ANTENNA2", q_idx, r0, nbl)
        ms.putcol("UVW", uvw, r0, nbl)
        ms.putcol("DATA", vis.astype(np.complex64), r0, nbl)
        flags = flag_bl[:, None, None] | (wspec <= 0)
        wspec[flags] = 0.0
        ms.putcol("FLAG", flags, r0, nbl)
        ms.putcol("WEIGHT_SPECTRUM", wspec.astype(np.float32), r0, nbl)
        wrow = wspec.sum(axis=1) / max(len(cidx), 1)                    # mean over observed channels
        with np.errstate(divide="ignore"):
            srow = np.where(wrow > 0, 1.0 / np.sqrt(wrow), 0.0)
        ms.putcol("FLAG_ROW", flag_bl, r0, nbl)
        ms.putcol("DATA_DESC_ID", np.full(nbl, ddid), r0, nbl)
        tt = t.mjd * 86400.0
        ms.putcol("TIME", np.full(nbl, tt), r0, nbl)
        ms.putcol("TIME_CENTROID", np.full(nbl, tt), r0, nbl)
        ms.putcol("INTERVAL", np.full(nbl, lst[0][2]), r0, nbl)
        ms.putcol("EXPOSURE", np.full(nbl, lst[0][2]), r0, nbl)
        ms.putcol("WEIGHT", wrow, r0, nbl)
        ms.putcol("SIGMA", srow, r0, nbl)
    for col in ("FIELD_ID", "ARRAY_ID", "OBSERVATION_ID", "PROCESSOR_ID", "SCAN_NUMBER",
                "STATE_ID", "FEED1", "FEED2"):
        ms.putcol(col, np.zeros(nrow, int))
    ms.close()

    print(f"wrote {args.ms}: {nant} antennas ({good.sum()} good), {nbl} baselines incl. autos, "
          f"layout '{args.layout}': {nspw} SPW(s), {len(blocks)} timestamps, {sum(len(s[2]) for s in sel)} channels ({sel[0][2][0][0]/1e6:.2f}-{sel[-1][2][-1][0]/1e6:.2f} MHz), "
          f"phase centre RA {np.degrees(ra0):.3f} Dec {np.degrees(dec0):.3f} deg; "
          f"element axes: {axes_src}")


if __name__ == "__main__":
    main()
