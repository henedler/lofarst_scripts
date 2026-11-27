#!/usr/bin/env python3
"""Calibrate one LOFAR2.0 XST measurement set (made with xst2ms.py), with the
ionospheric scintillation of Cas A and Cyg A taken out of the sky model first.

    calibrate_xst.py MS [--start N] [--stop N] [--nproc N] [--predict-args "..."] [--dry-run]

 1 predict      MODEL_CASA, MODEL_CYGA, MODEL_SKY (+ their sum MODEL_DATA_SUM), EEPs applied
 2 scalarphase  temporary per-dipole phases vs MODEL_DATA_SUM -> cal-scalarphase.h5,
                applied to DATA -> SCALARPHASE_DATA
 3 scint        per-snapshot scalar amplitude per direction (Cas A, Cyg A, diffuse sky),
                all dipoles constrained to one solution -> cal-scint.h5; plots-scint holds
                the scintillation amplitudes and the scalar phases applied before the solve
 4 scale        MODEL_CASA_SCALED = MODEL_CASA * g_CasA^2 / g_SKY^2, same for Cyg A (DP3 applycal);
                the diffuse sky does not scintillate, so g_SKY holds the instrument gain of the
                snapshot, which stays in the data for the bandpass
 5 sum          MODEL_DATA_SUM_SCALED = MODEL_CASA_SCALED + MODEL_CYGA_SCALED + MODEL_SKY
 6 imagemodel   debug image of MODEL_DATA_SUM_SCALED (I, V) -> img/img-model
 7 bandpass     diagonal solve on DATA vs MODEL_DATA_SUM_SCALED -> cal-bandpass.h5 (+ plots-bandpass)
 8 applybp      phase, then amplitude -> CORRECTED_DATA
 9 imagebp      img/img-bp
10 fulljones    full-Jones solve on CORRECTED_DATA -> cal-resonance.h5 (+ plots-fulljones)
11 applyfj      -> CORRECTED_DATA_FJ
12 imagefj      img/img-fj
13 resonance    losoto RESONANCE on cal-resonance.h5, two models (see RESONANCE_MODELS):
                  basic:    1 resonance, no background         -> amplitudeResBasic/phaseResBasic,
                            resonance_fit_basic.txt, plots-resonance-basic
                  advanced: 2 resonances + cubic background    -> amplitudeResAdvanced/phaseResAdvanced,
                            resonance_fit_advanced.txt, plots-resonance-advanced
                each plots dir: losoto plots of the model soltabs and resonance_dipoles.pdf
14 applyres     each model applied to CORRECTED_DATA (bandpass only) -> CORRECTED_DATA_RES_BASIC / _ADVANCED
15 imageres     img/img-res-basic, img/img-res-advanced
16 verifyfj     verification per model: full-Jones solve on its corrected data -> cal-verify-<model>.h5,
                plots-verify-<model> (losoto plots, verify_summary.png, verify_dipoles.pdf),
                verify_summary_<model>.txt
17 vstats       Stokes V flux statistics of img-model, img-bp, img-fj, img-res-basic and img-res-advanced
                with one common mask (elevation >= 10 deg, I significant in img-bp) -> img/vstats.txt

Every image step (imagemodel, imagebp, imagefj, imageres) also writes fractional circular polarisation
images V/I for every channel and the MFS image (img/<name>-<chan>-VoverI.fits and .png, colour scale
-0.5 ... +0.5, no blanking) and img/<name>-VoverI-overview.png.
Every command is printed (and written to the step log, together with its output).
Solutions, plots, images and logs go to cal_<MS name>/ next to the MS (or --outdir).
Needs DP3, wsclean, losoto (with the RESONANCE operation), python-casacore and lofstatcal.
"""
import argparse
import datetime
import os
import shlex
import shutil
import subprocess
import sys

import numpy as np
from casacore import tables as ct

SCRIPTS = os.path.dirname(os.path.realpath(__file__))
PARSETS = os.path.join(SCRIPTS, "parsets")

# resonance models fitted in step 13 and used in steps 14-17
RESONANCE_MODELS = {
    "basic": dict(parset="losoto-resonance-basic.parset", plot="losoto-plot-resonance-basic.parset",
                  soltabs=("amplitudeResBasic", "phaseResBasic"), fit="resonance_fit_basic.txt",
                  column="CORRECTED_DATA_RES_BASIC", image="img/img-res-basic"),
    "advanced": dict(parset="losoto-resonance-advanced.parset", plot="losoto-plot-resonance-advanced.parset",
                     soltabs=("amplitudeResAdvanced", "phaseResAdvanced"), fit="resonance_fit_advanced.txt",
                     column="CORRECTED_DATA_RES_ADVANCED", image="img/img-res-advanced"),
}

WSCLEAN_OPTS = ("-size 512 512 -scale 0.22 -pol IV -weight briggs 0.0 -channels-out 12 -fit-spectral-pol 7 "
                "-join-channels -mgain 0.9 -auto-mask 2.0 -auto-threshold 0.5 -nmiter 20 -niter 100000 "
                "-multiscale").split()


class Pipeline:
    def __init__(self, args):
        self.ms = os.path.realpath(args.ms.rstrip("/"))
        if not os.path.isdir(self.ms):
            sys.exit(f"no such MS: {self.ms}")
        name = os.path.basename(self.ms)
        name = name[:-3] if name.endswith(".ms") else name
        self.out = os.path.realpath(args.outdir or os.path.join(os.path.dirname(self.ms), "cal_" + name))
        self.args = args
        self.log = None
        os.makedirs(os.path.join(self.out, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.out, "img"), exist_ok=True)
        os.chdir(self.out)

    # ---- helpers
    def say(self, line):
        """Print a command to the terminal and to the step log."""
        print("    " + line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def run(self, *cmd):
        """Print and run a command, its output goes to the step log; stop the pipeline if it fails."""
        cmd = [str(c) for c in cmd]
        self.say("+ " + " ".join(shlex.quote(c) for c in cmd))
        if self.args.dry_run:
            return
        start = self.log.tell()
        ret = subprocess.run(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        if ret.returncode != 0:
            raise RuntimeError(f"{cmd[0]} exited with {ret.returncode}")
        if os.path.basename(cmd[0]) == "losoto":
            # losoto exits with 0 even if an operation is unknown or a step failed
            self.log.flush()
            with open(self.log.name) as f:
                f.seek(start)
                out = f.read()
            if "Unkown operation" in out or "Unknown operation" in out:
                raise RuntimeError("losoto does not know an operation of the parset (wrong losoto version in PATH? "
                                   "RESONANCE needs the losoto copy with resonance.py)")
            if "incomplete" in out and "losoto-plot" not in cmd[-1]:
                raise RuntimeError("a losoto step did not complete")

    def dp3(self, parset, **opts):
        self.run("DP3", os.path.join(PARSETS, parset), f"msin={self.ms}", *(f"{k}={v}" for k, v in opts.items()))

    def losoto(self, h5, parset, plotdir=None):
        if plotdir:
            self.say(f"+ rm -rf plots {plotdir}")
            if not self.args.dry_run:
                for d in ("plots", plotdir):
                    shutil.rmtree(d, ignore_errors=True)
        self.run("losoto", h5, parset)
        if plotdir:
            self.say(f"+ mv plots {plotdir}")
            if not self.args.dry_run:
                os.rename("plots", plotdir)

    def remove(self, path):
        self.say(f"+ rm -f {path}")
        if not self.args.dry_run and os.path.exists(path):
            os.remove(path)

    def image(self, name, column=None):
        """wsclean I+V image, then fractional circular polarisation V/I (FITS + PNG) for all channels and MFS"""
        self.run("wsclean", "-name", name, *(["-data-column", column] if column else []), *WSCLEAN_OPTS, self.ms)
        self.run(sys.executable, os.path.join(SCRIPTS, "fracpol_images.py"), name, "--vmax", "0.5")

    def antennas(self):
        return list(ct.table(self.ms + "/ANTENNA", ack=False).getcol("NAME"))

    def best_antenna(self):
        """antenna with the most unflagged visibilities"""
        t = ct.table(self.ms, ack=False)
        ok = (~t.getcol("FLAG")).sum(axis=(1, 2))
        counts = np.bincount(t.getcol("ANTENNA1"), weights=ok, minlength=len(self.antennas()))
        return self.antennas()[int(np.argmax(counts))]

    # ---- steps
    def predict(self):
        self.run(sys.executable, os.path.join(SCRIPTS, "predict_lfsm_ms.py"), self.ms,
                 "--column", "MODEL_DATA_SUM", "--processes", self.args.nproc, *shlex.split(self.args.predict_args))

    def scalarphase(self):
        self.remove("cal-scalarphase.h5")
        self.dp3("DP3-solve-scalarphase.parset", **{"sol.h5parm": "cal-scalarphase.h5"})
        self.dp3("DP3-correct.parset", **{"msin.datacolumn": "DATA", "msout.datacolumn": "SCALARPHASE_DATA",
                                          "cor.parmdb": "cal-scalarphase.h5", "cor.correction": "phase000"})

    def scint(self):
        self.remove("cal-scint.h5")
        self.dp3("DP3-solve-scint.parset", **{"sol.h5parm": "cal-scint.h5",
                                              "sol.antennaconstraint": "[[" + ",".join(self.antennas()) + "]]"})
        # all dipoles share the solution: plot one (the one with the most data), all directions per figure
        self.say("+ (write losoto-plot-scint.parset)")
        with open("losoto-plot-scint.parset", "w") as f:
            f.write("Ncpu = 0\n\n[plotScint]\noperation = PLOT\nsoltab = sol000/amplitude000\n"
                    f"axesInPlot = [time]\naxisInTable = dir\nant = [{self.best_antenna()}]\nprefix = plots/scint-\n")
        # plots-scint: the scalar phases applied before the solve, then the scintillation amplitudes
        self.say("+ rm -rf plots plots-scint")
        if not self.args.dry_run:
            for d in ("plots", "plots-scint"):
                shutil.rmtree(d, ignore_errors=True)
        self.run("losoto", "cal-scalarphase.h5", os.path.join(PARSETS, "losoto-plot-scalarphase.parset"))
        self.run("losoto", "cal-scint.h5", "losoto-plot-scint.parset")
        self.say("+ mv plots plots-scint")
        if not self.args.dry_run:
            os.rename("plots", "plots-scint")

    def scale(self):
        for d in ("CASA", "CYGA"):
            self.dp3("DP3-scale-model.parset", **{"msin.datacolumn": f"MODEL_{d}",
                                                  "msout.datacolumn": f"MODEL_{d}_SCALED",
                                                  "corrupt.parmdb": "cal-scint.h5", "corrupt.direction": f"[MODEL_{d}]",
                                                  "norm.parmdb": "cal-scint.h5"})

    def sum(self):
        # create the column as a copy of MODEL_SKY, then add the scaled compact sources
        self.run("DP3", f"msin={self.ms}", "msin.datacolumn=MODEL_SKY", "msout=.",
                 "msout.datacolumn=MODEL_DATA_SUM_SCALED", "steps=[]")
        query = "UPDATE $t SET MODEL_DATA_SUM_SCALED = MODEL_CASA_SCALED + MODEL_CYGA_SCALED + MODEL_SKY"
        self.say(f"+ taql '{query.replace('$t', self.ms)}'")
        if not self.args.dry_run:
            t = ct.table(self.ms, readonly=False, ack=False)  # noqa: F841 (used by taql as $t)
            ct.taql(query)
            t.close()

    def imagemodel(self):
        self.image("img/img-model", "MODEL_DATA_SUM_SCALED")

    def bandpass(self):
        self.remove("cal-bandpass.h5")
        self.dp3("DP3-solve-bandpass.parset", **{"msin.datacolumn": "DATA", "sol.h5parm": "cal-bandpass.h5",
                                                 "sol.modeldatacolumns": "[MODEL_DATA_SUM_SCALED]"})
        self.losoto("cal-bandpass.h5", os.path.join(PARSETS, "losoto-plot-bandpass.parset"), "plots-bandpass")

    def applybp(self):
        self.dp3("DP3-correct.parset", **{"cor.parmdb": "cal-bandpass.h5", "cor.correction": "phase000"})
        self.dp3("DP3-correct.parset", **{"msin.datacolumn": "CORRECTED_DATA",
                                          "cor.parmdb": "cal-bandpass.h5", "cor.correction": "amplitude000"})

    def imagebp(self):
        self.image("img/img-bp")

    def fulljones(self):
        self.remove("cal-resonance.h5")
        self.dp3("DP3-solveFJ.parset", **{"msin.datacolumn": "CORRECTED_DATA", "sol.h5parm": "cal-resonance.h5",
                                          "sol.modeldatacolumns": "[MODEL_DATA_SUM_SCALED]"})
        self.losoto("cal-resonance.h5", os.path.join(PARSETS, "losoto-plot-fj.parset"), "plots-fulljones")

    def applyfj(self):
        self.dp3("DP3-correct.parset", **{"msin.datacolumn": "CORRECTED_DATA", "msout.datacolumn": "CORRECTED_DATA_FJ",
                                          "cor.parmdb": "cal-resonance.h5", "cor.correction": "fulljones",
                                          "cor.soltab": "[amplitude000,phase000]", "cor.updateweights": "False"})

    def imagefj(self):
        self.image("img/img-fj", "CORRECTED_DATA_FJ")

    def resonance(self):
        for name, m in RESONANCE_MODELS.items():
            self.run("losoto", "cal-resonance.h5", os.path.join(PARSETS, m["parset"]))
            plotdir = f"plots-resonance-{name}"
            self.say(f"+ rm -rf plots {plotdir}")
            if not self.args.dry_run:
                for d in ("plots", plotdir):
                    shutil.rmtree(d, ignore_errors=True)
            self.run("losoto", "cal-resonance.h5", os.path.join(PARSETS, m["plot"]))
            self.run(sys.executable, os.path.join(SCRIPTS, "plot_resonance.py"), "cal-resonance.h5",
                     "--model", ",".join(m["soltabs"]), "--fit", m["fit"], "--out", "plots/resonance_dipoles.pdf")
            self.say(f"+ mv plots {plotdir}")
            if not self.args.dry_run:
                os.rename("plots", plotdir)

    def applyres(self):
        for m in RESONANCE_MODELS.values():
            self.dp3("DP3-correct.parset", **{"msin.datacolumn": "CORRECTED_DATA", "msout.datacolumn": m["column"],
                                              "cor.parmdb": "cal-resonance.h5", "cor.correction": "fulljones",
                                              "cor.soltab": "[" + ",".join(m["soltabs"]) + "]",
                                              "cor.updateweights": "False"})

    def imageres(self):
        for m in RESONANCE_MODELS.values():
            self.image(m["image"], m["column"])

    def verifyfj(self):
        for name, m in RESONANCE_MODELS.items():
            h5 = f"cal-verify-{name}.h5"
            self.remove(h5)
            self.dp3("DP3-solveFJ.parset", **{"msin.datacolumn": m["column"], "sol.h5parm": h5,
                                              "sol.modeldatacolumns": "[MODEL_DATA_SUM_SCALED]"})
            self.losoto(h5, os.path.join(PARSETS, "losoto-plot-fj.parset"), f"plots-verify-{name}")
            self.run(sys.executable, os.path.join(SCRIPTS, "plot_verify.py"), "cal-resonance.h5", h5,
                     "--model", ",".join(m["soltabs"]), "--outdir", f"plots-verify-{name}",
                     "--summary", f"verify_summary_{name}.txt")

    def vstats(self):
        self.run(sys.executable, os.path.join(SCRIPTS, "vstats.py"), "img/img-model", "img/img-bp", "img/img-fj",
                 *[m["image"] for m in RESONANCE_MODELS.values()], "--ref", "img/img-bp", "--ms", self.ms, "--min-elev", "10", "--snr", "5",
                 "--out", "img/vstats.txt")

    STEPS = ["predict", "scalarphase", "scint", "scale", "sum", "imagemodel", "bandpass", "applybp", "imagebp",
             "fulljones", "applyfj", "imagefj", "resonance", "applyres", "imageres", "verifyfj", "vstats"]

    def execute(self):
        print(f"MS:     {self.ms}\noutput: {self.out}")
        for n, name in enumerate(self.STEPS, start=1):
            if n < self.args.start or n > self.args.stop:
                continue
            logfile = os.path.join("logs", f"{n:02d}-{name}.log")
            print(f"[{datetime.datetime.now():%H:%M:%S}] step {n}: {name}  (log: {os.path.join(self.out, logfile)})",
                  flush=True)
            with open(logfile, "w") as self.log:
                try:
                    getattr(self, name)()
                except Exception as e:
                    self.log.write(f"\nFAILED: {e}\n")
                    self.log.flush()
                    with open(logfile) as f:
                        tail = f.readlines()[-25:]
                    sys.exit(f"  step {n} ({name}) FAILED: {e}\n  last lines of {logfile}:\n" + "".join(tail))
        print(f"[{datetime.datetime.now():%H:%M:%S}] done: {self.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ms")
    ap.add_argument("--start", type=int, default=1, help="first step (earlier steps must have run before)")
    ap.add_argument("--stop", type=int, default=len(Pipeline.STEPS), help="last step")
    ap.add_argument("--nproc", type=int, default=1, help="processes for the prediction (~3 GB each at nside 64)")
    ap.add_argument("--predict-args", default="", help='extra arguments for predict_lfsm_ms.py, e.g. "--bodies NONE"')
    ap.add_argument("--outdir", help="output directory (default: cal_<MS name> next to the MS)")
    ap.add_argument("--dry-run", action="store_true", help="only print the commands")
    Pipeline(ap.parse_args()).execute()


if __name__ == "__main__":
    main()
