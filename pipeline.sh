DP3 DP3-solve-bandpass.parset msin.datacolumn=DATA msin=L2086797_CS002LBA_XST.ms
losoto cal-bandpass.h5 losoto-plot-bandpass.parset
mv plots plots-bandpass
DP3 DP3-correct.parset msin=L2086797_CS002LBA_XST.ms cor.parmdb=cal-bandpass.h5 cor.correction=phase000
DP3 DP3-correct.parset msin=L2086797_CS002LBA_XST.ms cor.parmdb=cal-bandpass.h5 cor.correction=amplitude000 msin.datacolumn=CORRECTED_DATA
# image bandpass corrected data
wsclean -size 512 512 -scale 0.22 -pol IV -name img/img-bp -weight briggs 0.0  -channels-out 12 -fit-spectral-pol 7 -join-channels -mgain 0.9 -auto-mask 2.0 -auto-threshold 0.5 -nmiter 20 -niter 100000  -multiscale L2086797_CS002LBA_XST.ms
# leakage correction
DP3 DP3-solveFJ.parset msin.datacolumn=CORRECTED_DATA msin=L2086797_CS002LBA_XST.ms
losoto cal-fulljones.h5 losoto-plot-fj.parset
mv plots plots-fulljones
DP3 DP3-correct.parset msin=L2086797_CS002LBA_XST.ms msin.datacolumn=CORRECTED_DATA msout.datacolumn=CORRECTED_DATA_FJ \
               cor.parmdb=cal-fulljones.h5 cor.correction=fulljones cor.soltab=[amplitude000,phase000] \
               cor.updateweights=False
# image leakage corrected data
wsclean -size 512 512 -scale 0.22 -pol IV -name img/img-fj -data-column CORRECTED_DATA_FJ -weight briggs 0.0  -channels-out 12 -fit-spectral-pol 7 -join-channels  -mgain 0.9 -auto-mask 2.0 -auto-threshold 0.5 -nmiter 20 -niter 100000  -multiscale L2086797_CS002LBA_XST.ms
