# czerny_wd
<h2>Charles Zhu's Experimental (nuclear) RuNawaY of (possibly spinning, magnetic) White Dwarfs Code ([CZERNY](https://en.wikipedia.org/wiki/Carl_Czerny)-WD Code)</h2>

A Python-based hydrostatic spherical white dwarf (WD) integrator that utilizes Timmes' [Helmholtz EOS](http://cococubed.asu.edu/code_pages/eos.shtml).  Also included is effective pressure support and convective suppression due rotation (via the Chandrasekhar approximation and Solberg-Hoiland criterion), and convective suppression due to magnetic fields (using [Gough & Tayler 1966](http://adsabs.harvard.edu/abs/1966MNRAS.133...85G)).  Prototype was used to generate initial conditions for Zhu et al. 2013 and Zhu et al. 2015.  Currently being used to investigate nuclear runaways in CO WDs (Zhu et al. in prep.).

<h3>Requirements</h3>

I'm currently using the following modules:

- numpy 1.10.1
- scipy 0.16.1
- matplotlib 1.5.0
- cPickle 1.71

I don't guarantee that the code will work (though it likely will) for Python builds with earlier versions of these modules.  To install the modules above without overwriting your current Python setup, you can create a [Python virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/).

<h3>Building</h3>

To build, simply type

```
make
```

which builds myhelm_magstar.f90 using f2py.  The Makefile should only be used within the environment you'll be running the code!

<h3>Examples</h3>

All relevant functions have (or will soon have) docstrings associated with them that describe the functionality of their arguments.  Below are some common calculations performed with the code.

To build and plot the density profile of a cold 1.2 Msun CO WD with Omega = 0.3 s^-1:

```python
import StarMod as Star
import matplotlib.pyplot as plt
#Build a 1.2Msun star with solid body rotation Omega = 0.3 s^-1
mystar = Star.maghydrostar(1.2*1.9891e33, 5e6, False, omega=0.3, simd_userot=True, verbose=True)
plt.plot(mystar.data["R"], mystar.data["rho"], 'r-');plt.xlabel("r (cm)");plt.ylabel(r"$\rho$ (g/cm$^3$)")
```

To build a runaway track for a 1.2 Msun non-rotating, unmagnetized WD (warning; this will take about 10 minutes on a reasonable computer):

```python
import runaway as rw
import rhoTcontours as rtc
#Create runaway track using make_runaway()
static_tol = 1e-6
static_S = [1e7,2e7,3e7,4e7,5e7] + list(10.**arange(log10(6e7), log10(2.2e8*1.01), 0.005))
od_1pt2Msun = rw.make_runaway(starmass=1.2*1.9891e33, verbose=True, mass_tol=static_tol, S_arr=static_S, simd_userot=False, simd_usegammavar=False, simd_usegrav=False)
#Generates rho-T diagram
[fig, ax] = rtc.get_figure(axes=[10**5.5,10**9.25,10**7.0,10**9.75])
ax.plot(od_1pt2Msun["dens_c"], od_1pt2Msun["temp_c"], 'k-', lw=2)
```

To generate a series of adiabatic non-rotating WD profiles with increasing density:

```python
import StarMod as Star
import numpy as np
import copy
#Create density array
dens_c = 10.**np.arange(8,9,0.1)
#Store outputs
out_dict = {"dens_c": dens_c,
	"M": np.zeros(len(dens_c)),
	"stars": []}
#Generate dummy (some conditional flags are star specific, and not generally changed at the integrate_star level; see maghydrostar.__init__() )
#stop_mrat cancels integration when m > self.mass_want*stop_mrat and is normally set to 2; we're not inputting a mass here, so flag that as false.
mystar = Star.maghydrostar(False, False, False, simd_userot=True, verbose=True, stop_mrat=False, dontintegrate=True)
#Loop over density array
for i in range(len(dens_c)):
	[Mtot, outerr_code] = mystar.integrate_star(dens_c[i], 5e6, 0.0, recordstar=True, outputerr=True)
	out_dict["M"][i] = Mtot
	out_dict["stars"].append(copy.deepcopy(mystar.data))
```
 
