# lofarst_scripts

Tools to process LOFAR2.0 single-station XST (cross-correlation statistics) data with the standard interferometric tools DP3, WSClean and LoSoTo.

## Documentation

- [Investigation of the LBA element resonance](docs/lba_element_resonance.md): what these tools were developed for. The write-up for the LOFAR Beam workshop covers the data, the sky model, the calibration pipeline, the resonance fit, the results and the open questions.

## Main tools

### `xst2ms.py`: XST to Measurement Set

Converts a LOFAR2.0 XST statistics file (HDF5) into a Measurement Set.

```
python xst2ms.py XST.h5 OUT.ms [--fmin 30] [--fmax 67] [--ra DEG --dec DEG] [--time-offset 0] [--layout single|spw] [--no-conjugate]
```

- **Antennas:** each antenna (dipole pair) becomes an MS antenna and each pair of antennas a baseline, autocorrelations included.
- **Timing:** the XST sweep measures every subband once, at its own 10 s timestamp. With `--layout single` (default) there is one spectral window with all channels, and each timestamp holds data only in its own channels; DP3 and EveryBeam need this. `--layout spw` writes one spectral window per timestamp, which is more compact and works with WSClean only.
- **Phase centre:** the raw, zenith-referenced visibilities are phase-rotated to one fixed J2000 phase centre, by default the zenith at the middle of the sweep, and the UVWs are computed for it.
- **Sign convention:** the visibilities are complex-conjugated by default, so that WSClean images the sky correctly. `--no-conjugate` keeps the raw XST sign.
- **Weights:** `WEIGHT_SPECTRUM` comes from the autocorrelations via the radiometer equation. Visibilities with a non-positive autocorrelation are flagged.
- **EveryBeam:** the MS is marked as AARTFAAC data (`TELESCOPE_NAME = AARTFAAC`, antenna type LBA), so EveryBeam treats every antenna as one LBA dipole.

Requires numpy, h5py, python-casacore and astropy.

### `predict_lfsm_ms.py`: sky-model prediction with embedded element patterns

Predicts the model visibilities for an MS made with `xst2ms.py`, using the sky and antenna model of [lofar_station_calibration](https://git.astron.nl/mckenna/lofar_station_calibration) (lofstatcal) by David McKenna.

```
python predict_lfsm_ms.py MS [--split-columns ...] [--column MODEL_DATA_SUM] [--model-source EEP]
                             [--sources "Cas A,Cyg A"] [--bodies SUN] [--sky-nside 64]
                             [--elevation-cutoff 5] [--processes 1] [--no-conjugate]
```

- **Sky model:** the diffuse sky from the LWA Low Frequency Sky Model (pygdsm), plus Cas A and Cyg A (default) and the Sun as separate sources.
- **Antenna model:** the per-antenna LOBES embedded element patterns (EEPs) are applied to every sky pixel and source (`--model-source EEP`, default). Other lofstatcal models can be chosen, e.g. `EB_LOBES` or `EB_HAMAKER`.
- **Output columns:** by default one column per sky component, for direction-dependent solves: `MODEL_CASA`, `MODEL_CYGA` and `MODEL_SKY`. The diffuse sky has the compact sources cut out, so each part of the sky is in exactly one column; unmapped components such as the Sun go into `MODEL_SKY`. `--column NAME` also writes the sum of all components; `--split-columns NONE` writes only the sum.
- **Conventions:** the model goes through the same phase rotation and conjugation as the data in `xst2ms.py`. Use `--no-conjugate` if the MS was made with `--no-conjugate`.
- **Memory:** about 3 GB per process at `--sky-nside 64`.

Requires lofstatcal (with its EEP files and interpolator cache set up) and python-casacore.

## Other scripts

- `calibrate_xst.py`: the full calibration pipeline for one MS, from prediction and scintillation correction to bandpass, full-Jones and resonance fits, verification and Stokes V statistics. See the [documentation](docs/lba_element_resonance.md#calibration-experiment).
- `parsets/`: the DP3 and LoSoTo parsets used by the pipeline.
- `fracpol_images.py`, `vstats.py`, `plot_resonance.py`, `plot_verify.py`: helpers for the pipeline's V/I images, Stokes V statistics and diagnostic plots.

The resonance fit uses the `RESONANCE` operation in a modified LoSoTo.
