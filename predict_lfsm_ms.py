#!/usr/bin/env python3
"""Predict MODEL_DATA in an XST measurement set (made with xst2ms.py) with the
sky and antenna model of lofar_station_calibration (lofstatcal).

The model is computed exactly as in lofstatcal's calibration
(lofstatcal.visibility.model.worker_generate_corrected_visibilities):
  - diffuse sky from the LWA Low Frequency Sky Model (pygdsm), healpix sky_nside,
    with the compact sources cut out (simulate_station_visibilities /
    generate_vis_from_diffuse_background),
  - Cas A and Cyg A (default) plus the Sun as separate sources with the
    de Gasperin et al. (2020) spectra (generate_vis_from_sources). lofstatcal's
    solar spectrum, 1.94e-4 f[MHz]^1.992, is in solar flux units, so it is
    converted to Jy (x 1e4) via lofstatcal's solar_flux_scaling,
  - per-antenna embedded element patterns applied to every sky pixel/source
    (generate_and_apply_eeps), by default lofstatcal's own EEP interpolators,
    which are built and cached on disk on the first run and re-used afterwards,
  - lofstatcal's station geometry (lofarantpos), baselines and antenna order.

The resulting raw-convention coherency matrices (96x96x2x2 per channel, one
per XST subband at its own snapshot time) are then
  1. converted to Jy and to the MS convention I = (XX+YY)/2:
     lofstatcal works in Kelvin per sky pixel with the flux split over both
     polarisations, so the model is multiplied by 2 x (Jy per K of a pixel),
  2. put through exactly the transform xst2ms.py applies to the data: phase
     rotation to the MS phase centre with the MS UVWs, and complex conjugation
     (default; --no-conjugate if the MS was made with --no-conjugate).

Only the channels that hold data in a snapshot are predicted. Fully flagged
rows get a zero model.

Output columns (one per sky component, for direction-dependent solves, e.g.
DP3 ddecal with modeldatacolumns=[MODEL_CASA,MODEL_CYGA,MODEL_SKY]):
  --split-columns "Cas A=MODEL_CASA,Cyg A=MODEL_CYGA,SKY=MODEL_SKY" (default)
maps lofstatcal components to MS columns. SKY is the diffuse LFSM sky, from
which every modelled compact source is cut out (pixels within 3 deg replaced
by the mean within 5 deg, lofstatcal's subtract_compact_sources_from_pygdsm_map),
so each part of the sky is in exactly one column. Components that are not
mapped (e.g. the Sun) are added to the SKY column. Every component is
multiplied by the EEPs separately. --column NAME additionally writes the sum
of all components; --split-columns NONE writes only that sum (old behaviour).

lofstatcal cuts a compact source out of the diffuse map only while it is
above object_elevation_cutoff (default 20 deg in lofstatcal); below that it
would stay in the LFSM and land in the SKY column. This script therefore sets
the source and diffuse-sky elevation cutoffs to the same value
(--elevation-cutoff, default 5 deg), so a source is either its own component
or below the horizon of the model.

Usage:
    predict_lfsm_ms.py MS [--split-columns ...] [--column MODEL_DATA] [--model-source EEP]
                          [--sources "Cas A,Cyg A"] [--bodies SUN] [--sky-nside 64]
                          [--no-diffuse-sky] [--processes 1] [--no-conjugate]

The EEP model needs the same set-up as lofstatcal itself: EEP files in EEP_DIR
(or lofstatcal's default location), interpolator cache in INTERP_CACHE_DIR.
Requires lofstatcal (pip install of lofar_station_calibration) and python-casacore.
"""
import argparse
import datetime
import sys

import numpy as np
from casacore import tables as ct

C_LIGHT = 299792458.0
SB_WIDTH = 100e6 / 512                     # LBA (10_90 / 30_90), 200 MHz clock
SFU_TO_JY = 1e4                            # 1 solar flux unit = 1e4 Jy


def jy_per_kelvin(pixel_area_deg2, freq_MHz):
    """Flux density (Jy) of a pixel of the given area per Kelvin of brightness temperature.
    lofstatcal's kelvin_to_Jy(x, area, f) returns x divided by this factor, so we use its inverse."""
    from lofstatcal.misc.sources import kelvin_to_Jy
    return 1.0 / kelvin_to_Jy(1.0, pixel_area_deg2, freq_MHz)


def ensure_column(ms, column, comment="lofstatcal sky x EEP model (predict_lfsm_ms.py)"):
    if column in ms.colnames():
        return
    desc = ms.getcoldesc("DATA")
    desc["comment"] = comment
    ms.addcols(ct.maketabdesc(ct.makecoldesc(column, desc)))
    print(f"added column {column}")


def mjd_seconds_to_datetime(t):
    return datetime.datetime(1858, 11, 17) + datetime.timedelta(seconds=float(t))


def run_worker(config, inputs, worker_id):
    from lofstatcal.visibility.model import worker_generate_corrected_visibilities
    return worker_generate_corrected_visibilities(config, inputs, worker_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ms")
    ap.add_argument("--split-columns", default="Cas A=MODEL_CASA,Cyg A=MODEL_CYGA,SKY=MODEL_SKY",
                    help="component=COLUMN pairs; SKY = diffuse sky + all unmapped components; NONE = no split")
    ap.add_argument("--column", default=None,
                    help="also write the sum of all components to this column (default: MODEL_DATA if "
                         "--split-columns NONE, otherwise not written)")
    ap.add_argument("--elevation-cutoff", type=float, default=5.0,
                    help="elevation (deg) below which neither compact sources nor diffuse-sky pixels are modelled")
    ap.add_argument("--station", help="station name, e.g. CS002 or CS002LBA (default: from the MS)")
    ap.add_argument("--antenna-set", default="LBA")
    ap.add_argument("--antenna-band", default="10_90", help="filter, for lofstatcal's subband frequencies")
    ap.add_argument("--model-source", default="EEP",
                    help="lofstatcal antenna model: EEP (default), EB_LOBES, EB_HAMAKER, EB_UNITARY, DB_HAMAKER")
    ap.add_argument("--no-eeps", action="store_true", help="no antenna response at all (use_eeps=False)")
    ap.add_argument("--sources", default="Cas A,Cyg A",
                    help="comma-separated lofstatcal source names (Cas A, Cyg A, Tau A, Vir A) or NONE")
    ap.add_argument("--bodies", default="SUN", help="comma-separated solar-system bodies (SUN, MOON, JUPITER) or NONE")
    ap.add_argument("--sky-nside", type=int, default=64,
                    help="healpix resolution of the diffuse sky (default 64 = native grid of the 5.1 deg LFSM; lofstatcal default 32; ~3 GB per process)")
    ap.add_argument("--no-diffuse-sky", action="store_true", help="only the compact sources, no LFSM")
    ap.add_argument("--single-antenna-sampling", default=-1,
                    help="lofstatcal single_antenna_sampling (-1: every antenna its own EEP)")
    ap.add_argument("--processes", type=int, default=1,
                    help="parallel workers; each builds/loads its own interpolators (memory!)")
    ap.add_argument("--conjugate", action=argparse.BooleanOptionalAction, default=True,
                    help="conjugate like xst2ms.py does by default (use --no-conjugate to match an MS made with it)")
    args = ap.parse_args()

    from lofstatcal.configuration.calibration_config import generate_default_calibration_configuration
    from lofstatcal.misc.stations import get_station_baselines, get_station_earthlocation
    from lofstatcal.misc.freq_and_sb import frequency_from_subband

    split = lambda s: [x.strip() for x in s.split(",") if x.strip() and x.strip().upper() != "NONE"]
    config, unparsed = generate_default_calibration_configuration({
        "source_names": split(args.sources),
        "body_names": split(args.bodies),
        "model_source": args.model_source,
        "use_eeps": not args.no_eeps,
        "sky_nside": args.sky_nside,
        "use_diffuse_sky": not args.no_diffuse_sky,
        "single_antenna_sampling": args.single_antenna_sampling,
        "solar_flux_scaling": SFU_TO_JY,                                   # lofstatcal's Sun is in SFU
        "object_elevation_cutoff": args.elevation_cutoff,                  # same horizon for sources and
        "diffuse_sky_elevation_cutoff": args.elevation_cutoff,             # diffuse sky (see docstring)
    })

    # ---- output columns: component name -> column
    colmap = {}
    for item in split(args.split_columns):
        if "=" not in item:
            raise SystemExit(f"--split-columns: expected component=COLUMN, got {item!r}")
        comp, col = (x.strip() for x in item.split("=", 1))
        colmap[comp] = col
    if colmap:
        known = set(config["source_names"]) | {b.upper() for b in config["body_names"]} | {"SKY"}
        unknown = [c for c in colmap if c not in known and c.upper() not in known]
        if unknown:
            raise SystemExit(f"--split-columns: {unknown} are not modelled components ({sorted(known)})")
        if "SKY" not in colmap:
            raise SystemExit("--split-columns needs a SKY=COLUMN entry (diffuse sky and unmapped components)")
        if "SKY" in colmap and args.no_diffuse_sky:
            print("note: --no-diffuse-sky, so the SKY column only holds unmapped compact components (if any)")
    total_column = args.column if args.column else (None if colmap else "MODEL_DATA")
    out_columns = list(dict.fromkeys(list(colmap.values()) + ([total_column] if total_column else [])))

    ms = ct.table(args.ms, readonly=False, ack=False)
    ant = ct.table(args.ms + "/ANTENNA", ack=False)
    nant = ant.nrows()
    station = (args.station or ant.getcell("STATION", 0)).upper()
    if station.endswith(args.antenna_set.upper()):                         # accept CS002LBA as well as CS002
        station = station[: -len(args.antenna_set)]
    field = f"{station}{args.antenna_set.upper()}"
    baselines = get_station_baselines(field)
    location = get_station_earthlocation(field)
    if baselines.shape[0] != nant:
        raise SystemExit(f"lofstatcal has {baselines.shape[0]} antennas for {field}, the MS has {nant}")
    spw = ct.table(args.ms + "/SPECTRAL_WINDOW", ack=False)
    ddesc = ct.table(args.ms + "/DATA_DESCRIPTION", ack=False).getcol("SPECTRAL_WINDOW_ID")
    for col in out_columns:
        ensure_column(ms, col, "lofstatcal sky x EEP model (predict_lfsm_ms.py)" if col == total_column else
                      "lofstatcal model of " + ", ".join(k for k, v in colmap.items() if v == col) + " x EEP (predict_lfsm_ms.py)")

    # ---- one lofstatcal input per (snapshot, channel with data)
    times = ms.getcol("TIME")
    dds = ms.getcol("DATA_DESC_ID")
    blocks, inputs, where = [], [], []
    for tval, dd in sorted(set(zip(times, dds))):
        rows = np.where((times == tval) & (dds == dd))[0]
        r0, nrow = rows[0], rows[-1] - rows[0] + 1
        assert nrow == len(rows), "rows of a block must be contiguous"
        flags = ms.getcol("FLAG", r0, nrow)
        chans = np.where(~flags.all(axis=(0, 2)))[0]
        row_ok = ~flags.all(axis=(1, 2))
        freqs = spw.getcell("CHAN_FREQ", int(ddesc[dd]))
        interval = float(ms.getcell("INTERVAL", r0))
        centre = mjd_seconds_to_datetime(tval)                              # MS TIME = snapshot centre
        ib = len(blocks)
        blocks.append((r0, nrow, row_ok, freqs))
        for c in chans:
            isub = int(round(freqs[c] / SB_WIDTH))
            f_sb = frequency_from_subband(isub, args.antenna_band)
            if abs(f_sb - freqs[c]) > 1.0:
                raise SystemExit(f"channel {freqs[c]} Hz is not subband {isub} ({f_sb} Hz) of band {args.antenna_band}")
            metadata = {
                "station_name": station,
                "antenna_set": args.antenna_set.upper(),
                "antenna_band": args.antenna_band,
                "antenna_subband": isub,
                "pointing": None,                                            # LBA: no beamformer pointing
                "timestamp": (centre - datetime.timedelta(seconds=interval / 2)).strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "integration_interval": interval,
                "integration_centre": centre,
            }
            inputs.append({
                "isub": isub,
                "cube": np.zeros((nant, nant), dtype=np.complex64),
                "metadata": metadata,
                "baselines": baselines,
                "station_earthlocation": location,
                "calibration_configuration": config,
            })
            where.append((ib, c))

    print(f"{args.ms}: {field}, {len(blocks)} snapshots, {len(inputs)} channels to predict; "
          f"model_source={config['model_source'] if config['use_eeps'] else 'none'}, "
          f"diffuse={config['use_diffuse_sky']} (nside {config['sky_nside']}), "
          f"sources={config['source_names']}, bodies={config['body_names']}, "
          f"elevation cutoff {args.elevation_cutoff} deg")
    print("output: " + ", ".join([f"{k} -> {v}" for k, v in colmap.items()] +
                                 ([f"sum -> {total_column}"] if total_column else [])))

    # ---- lofstatcal model (its own tqdm progress bars)
    if args.processes > 1:
        import joblib
        chunks = [list(range(i, len(inputs), args.processes)) for i in range(args.processes)]
        parts = joblib.Parallel(n_jobs=args.processes)(
            joblib.delayed(run_worker)(config, [inputs[k] for k in idx], i + 2) for i, idx in enumerate(chunks))
        results = [None] * len(inputs)
        for idx, part in zip(chunks, parts):
            for k, r in zip(idx, part):
                results[k] = r
    else:
        results = run_worker(config, inputs, 2)

    # ---- units, MS convention, write
    pixel_area = None
    if config["use_diffuse_sky"]:
        import healpy
        pixel_area = healpy.nside2pixarea(config["sky_nside"], degrees=True)
    a1_all = ms.getcol("ANTENNA1")
    a2_all = ms.getcol("ANTENNA2")
    uvw_w = ms.getcol("UVW")[:, 2]

    def column_of(comp):
        if comp in colmap:
            return colmap[comp]
        if comp.upper() in colmap:
            return colmap[comp.upper()]
        return colmap["SKY"]

    by_block = {}
    for (ib, c), res in zip(where, results):
        by_block.setdefault(ib, []).append((c, res))
    seen = {}                                                                # component -> number of channels
    for ib, (r0, nrow, row_ok, freqs) in enumerate(blocks):
        out = {col: np.zeros((nrow, len(freqs), 4), dtype=np.complex64) for col in out_columns}
        a1 = a1_all[r0:r0 + nrow][row_ok]
        a2 = a2_all[r0:r0 + nrow][row_ok]
        for c, res in by_block.get(ib, []):
            f = freqs[c]
            # lofstatcal: Kelvin per pixel (diffuse sky on) or Jy (off), flux split over the two pols
            scale = 2.0 * (jy_per_kelvin(pixel_area, f / 1e6) if pixel_area else 1.0)
            rot = np.exp(2j * np.pi * uvw_w[r0:r0 + nrow][row_ok] * f / C_LIGHT)[:, None]   # as xst2ms
            for comp, R in res["model_vis_sources"].items():                 # R: (nant, nant, 2, 2), raw convention
                seen[comp] = seen.get(comp, 0) + 1
                v = scale * R[a1, a2].reshape(-1, 4) * rot                   # XX, XY, YX, YY
                if args.conjugate:
                    v = np.conj(v)
                if colmap:
                    out[column_of(comp)][row_ok, c] += v
                if total_column:
                    out[total_column][row_ok, c] += v
        for col in out_columns:
            ms.putcol(col, out[col], r0, nrow)
    ms.close()
    nch = len(where)
    for comp, n in sorted(seen.items()):
        print(f"  {comp}: in {n}/{nch} channels -> {column_of(comp) if colmap else total_column}")
    for comp in colmap:
        if comp != "SKY" and not any(k == comp or k.upper() == comp.upper() for k in seen):
            print(f"  WARNING: {comp} was never above {args.elevation_cutoff} deg: {colmap[comp]} is zero")
        elif comp != "SKY" and seen.get(comp, seen.get(comp.upper(), 0)) < nch:
            print(f"  note: {comp} is below {args.elevation_cutoff} deg for part of the observation "
                  f"(zero model there)")
    print("wrote " + ", ".join(out_columns))


if __name__ == "__main__":
    sys.exit(main())
