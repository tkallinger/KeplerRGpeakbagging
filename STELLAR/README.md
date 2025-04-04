# STELLAR: <ins>S</ins>olar-<ins>T</ins>yp<ins>E</ins> osci<ins>LL</ins>ation <ins>A</ins>nalyse<ins>R</ins>

STELLAR needs ([UltraNest](https://johannesbuchner.github.io/UltraNest/index.html)), which is installed with 
```
pip install ultranest
```
Then just copy the files <stellar.py> and <RFmodel_dnu.pkl> to your prefered folder.
***
STELLAR is a Python class that **automatically** performs a comprehensive analysis of any solar-type oscillating star (MS to AGB). With an power density spectrum from any source (Kepler, TESS, ...) as only input, it determines the granulation background, the global properties of the power excess (fmax, dnu, dnu02, ...), the evolutionary stage (MS, RGB, RC, ABG), and finally all significant l=0 to 3 modes (including rotationaly splittings for MS stars). 

A STELLAR run is initialised as
```
from stellar import stellar
star = stellar(ID, pds, <path='files'>, <f_nyg=4165.2>, <verbose=True>)
```
where:
- *ID*: project name
- *pds*: pandas Dataframe with columns *f* (frequency in microHz) and *p* (power density in ppm^2/microHz)
- *path*: if not existing, a folder with name *ID* is created in the folder *path*, where all output of STELLAR is stored.
- *f_nyg*: Nyquist frequency of the power density spectrum. Default value is the 2min TESS observations.
- *verbose*: to supress the plenty of UltraNest output set it to *False*

STELLAR has the following class methods (which build on each other - so keep the order):
***
### Granulation background fit
```
star.fmax_fitter(<fmax_guess=False>, <plot=False>)
```
The *fmax_fitter* automatically finds the approximate position of the power excess and then fits a global model to the power density spectrum following the approach of [Kallinger et al. (2014)](https://ui.adsabs.harvard.edu/abs/2014A%26A...570A..41K/abstract). In rare cases, finding the approximate position fails, and an initial guess *fmax_guess* is needed. *fmax_fitter* also determines if the power excess is statistically significant. Fitting parameters are stored in <*ID.bg_par.dat*> and the fit is plotted in <*ID.pdf*> if *plot* is set True. 
***
### Central large frequency separation
```
star.dnu_fitter(<dnu_guess=False>, <flip_fc=False>, <plot=False>)
```
The *dnu_fitter* uses the global fit parameters to predict initial guesses for the large and small frequency separation *dnu* and *dnu02* from a random forest regressor, which is trained on data of 6000+ Kepler red giants and accurate to about 2%. This works only for stars with *fmax* smaller than about 280 microHz. For star with a larger *fmax*, *dnu_guess* needs to be provided (in a later version, I will implement an estimator for this as well).\
The *dnu_fitter* then determines *dnu* and *dnu02* in the central three radial orders around *fmax* and the frequency of the central radial mode *f_c* following [Kallinger et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010A%26A...509A..77K/abstract). Fitting parameters are stored in <*ID*.dnu_par.dat> and the fit is plotted in <*ID*.pdf> if *plot* is set True.\
Based on *dnu* and *f<sub>c</sub>*, the evolutionary stage of red giants is determined following [Kallinger et al. 2012](https://ui.adsabs.harvard.edu/abs/2012A%26A...541A..51K/abstract).\
Sometimes, especially for MS stars, initialising *f_c* for the fit fails and a neighbouring dipole mode is miss-identified as the central radial modes. In such rare cases, setting *flip_fc* to True corrects for this.
***
### Peakbagging of l=0 and 2 modes
```
star.peakbag_02(<alpha=None>, <l1_threshold=8>, <odds_ratio_limit = 5>, <rotation=False>, <incl_prior=None>, <plot=False>, <log=False>)
```
The peakbagging methods are based on the original version of ABBA ([Kallinger 2019](https://ui.adsabs.harvard.edu/abs/2019arXiv190609428K/abstract)) but with a few improvements and expansion.\

Prerequisites are *dnu*, *dnu02*, and *f<sub>c</sub>* from the *dnu_fitter* and the global properties of the power excess (*fmax* and *sig*). *peakbag_02* performs the following steps to find all significant l = 0 and 2 modes in the spectrum:
- The number of searched radial orders is [-dn,0,dn] relative to the order of *f<sub>c</sub>*, where *dn = round(*3sig*/*dnu*) + 1*. The position of the radial modes is estimated as *f<sub>n</sub> = f<sub>c</sub> + n*(1 + alpha/2*n<sup>2</sup>), where the curvature parameter *alpha* is determined from a scaling law (based on the 6000+ Kepler red giants).
- Two Lorentzian profiles are then fitted to the range [*-1.5 dnu02, +0.5 dnu02*] around *f<sub>n</sub>*. If *l1_threshold* is not *None*, the residual spectrum is checked for narrow peaks (due to diploe modes in the vicinity of the l = 0 and 2 modes) exceeding the signal-to-background ratio threshold. IF one or more are found, they are supressed and the fit is redone.
- The individual modes need to exceed the given *odds_ratio_limit* to be accepted as stastically significant mode, where the odds_ration is defined as the probility ratio with and without the fit that result from the global evidence of the fit.
- If *rotation* is set to *True*, the l = 2 mode includes rotational split components, which are parameterised by the rotational split frequency *f<sub>rot</sub>* and an *inclination* angle. A Gaussian prior can be set for the inclination with *incl_prior=[a,b]*, where *a* and *b* are the expected value and its uncertainty, respectively. Usually, a first run is done without a prior, then the mean value of the individual inclinations is computed, which then serves as a prior for a 2nd run. This significantly improves the uncertainties of *f<sub>rot</sub>*.
- If the modes are found to be significant, the fit is prewithened from the power density spectrum.
- Finally, the curvature of the l = 0 modes is fitted acording to [Kallinger et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...616A.104K/abstract).

The main output (all frequency parameters, ...) is stored in a file <*ID*.modes.dat>. For more detailed output, set *log* to *True*. If *plot* is set *True*, various plots are produced.

### Peakbagging of l=1 and 3 modes
```
star.peakbag_13(<ev_limit = 0.8>, <snr_limit=7.5>, <iter_limit=150>, <plot=False>, <rotation=False>, <incl_prior=None>)
```
*peakbag_13* performs the following steps to find all significant l = 1 and 3 modes in the spectrum:
- Search for the largest amplitdue peak in the residual (i.e., free of any l = 0 and 2 modes) spectrum and fit a Lorentzian profile to a window centered on this peak. The window size depends on if pure pressure modes (in MS stars) or mixed dipole modes (in red giants) are expected and adopts to the actual properties of the dipole modes spectrum. 
- If the resulting mode linewidth is compareable to the resolution limit (i.e. the maximum mode lifetime that can be resolved with the given observing lenght, ~480d in Kepler LC data), the fit is redone with a squared sinc function.
- The mode significance is rated by comparing the global fit evidences and the pure background signal (i.e., no mode) and needs to exceed the given *ev_limit* to be acceppted. If the mode is accepted, the best fit is prewithened from the residual power density spectrum. 
- Rotation works as for the l = 2 modes in the method *peakbag_02* but is currently only recommended for MS and sub giant stars.
- The procedure is prepeated until one of the termination criteria (*ev_limit, snr_limit, iter_limit*, where the latter simply counts the number of modes fitted) is given. 

Once all significant modes are extraced, the l = 3 modes are idetified amoung them according to their position in the frequency spectrum and their reltive amplitdue.\

The mode parameters are added to the file <*ID*.modes.dat> and plots are updated if *plot* is set to *True*.

The method only works for stars with *fmax > 30 microHz* (hard limit in the code) as for stars with smaller *fmax*, the l=1 modes are too densly packed to be resolved, even in the 4 yr long Kepler timeseries. **The peakbaging of dipole modes is still in an experimental phase as it is quite challenging to automate this for the large variety of possibilities.**

***
I also plan to add a further methods:
- automatically identify rotational splittings in the mixed dipole spectrum of red giants and determine the corresponding core rotation rate according to a new (dP1 independent) method (Schauer & Kallinger, in prep.).
- automatically identify dP1.


For further informations please contact tkallinger@me.com
