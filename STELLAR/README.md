# STELLAR: <ins>S</ins>olar-<ins>T</ins>yp<ins>E</ins> osci<ins>LL</ins>ation <ins>A</ins>nalyse<ins>R</ins>

STELLAR needs ([UltraNest](https://johannesbuchner.github.io/UltraNest/index.html)) Install it with 
```
pip install ultranest
```
STELLAR **automatically** performs a comprehensive analysis of any solar-type oscillating star (MS to AGB). With an power density spectrum from any source (Kepler, TESS, ...) as only input, it determines the granulation background, the global properties of the power excess (fmax, dnu, dnu02, ...), the evolutionary stage (MS, RGB, RC, ABG), and finally all significant l=0 to 3 modes (including rotationaly splittings for MS stars). 


A library of frequencies, amplitdes, and lifetimes of more than 250,000 individual l=0 to 3 oscillations modes of 6,179 red giants from APOKASC sample ([Pinsonneault et al. 2018](https://ui.adsabs.harvard.edu/abs/2018ApJS..239...32P/abstract)), which were extracted with the **A**utomated **B**ayesian peak-**B**agging **A**lgorithm (ABBA).

Visualisations (two pdf files per star) of the results are located in the directories [EchellePlots](https://github.com/tkallinger/KeplerRGpeakbagging/tree/master/EchellePlots) and [SpectrumPlots](https://github.com/tkallinger/KeplerRGpeakbagging/tree/master/SpectrumPlots)

The [modefiles](https://github.com/tkallinger/KeplerRGpeakbagging/tree/master/ModeFiles) contains general information about the star in the header:
- *fmax*: The frequency of the maximum oscillation power in microHz as defined in Eq.2 in [Kallinger et al. (2014)](https://ui.adsabs.harvard.edu/abs/2014A%26A...570A..41K/abstract) with two super-Lorentzian functions with a fixed exponent of four. The 1 sigma uncertainty is given as fmax_e.
- *dnu*, *dnu02*, and *f_c*: The large and small frequency separation determined in the central three radial orders around fmax and the frequency of the central radial mode as defined in Eq.2 of [Kallinger et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010A%26A...509A..77K/abstract). All parameters are in microHz.
- *dnu_cor* and *alpha*: Curvature-corrected large separation in microHz and the corresponding dimensionless curvature parameter as defined in Eq.,4 of [Kallinger et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...616A.104K/abstract).
- *evo*: Evolutionary stage of the star determined from the phase shift of the central radial mode ([Kallinger et al. 2012](https://ui.adsabs.harvard.edu/abs/2012A%26A...541A..51K/abstract)) with the following code: 0 - RGB star, 1 - RC star, 2 - secondary clump star, and 3 - AGB star.

The individual mode parameters are given in the main block:
- *l*: Mode degree
- *freq*: Mode frequency in microHz
- *amp*: Rms amplitude of the mode in ppm (i.e. the square root of the integrated power density under the mode profile)
- *tau*: Mode lifetime in days (0 for unresolved modes fitted with a squared sinc-function)
- *ev* and *ev1*: mode evidence (i.e. the probability that the mode is not due to noise). A useful threshold is 0.91, which corresponds to *strong evidence* in probability theory. The *ev1* parameter is only valid for l=1 modes. 

# Usage
To download the full repository (~650MB) type 
```
git clone https://github.com/tkallinger/KeplerRGpeakbagging.git
```
Individual modefiles can be downloaded using the Python function [read_modefile.py](https://github.com/tkallinger/KeplerRGpeakbagging/blob/master/read_modefile.py)

A summary of global seismic parameters of all stars in the library is given in [summary.dat](https://github.com/tkallinger/KeplerRGpeakbagging/blob/master/summary.dat)

More details may be found in [arXiv:1906.09428](https://arxiv.org/abs/1906.09428)

For further informations please contact tkallinger@me.com
