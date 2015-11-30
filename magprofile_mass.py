import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import os, sys
import copy

################HYDROSTAR CLASS###############################

class magprofile:
	"""Magnetic field profiler.  Returns functions fBfld and fBderiv, which allow for 

		Arguments:
		radius - radius
		mass - mass
		Bfld - B(r) profile
		gamma - defined in Feiden & Chaboyer 12 as a geometric correction to the magnetic pressure, i.e. Pchi = (gamma - 1)B^2/8/pi.  Defaults to 4./3. (maximum magnetic tension), can go up to 2 (no tension)
		filename - read profile from file (class constructor will not read radius or Bfld in that case)
		getpres - if non-zero, the gamma value of (gamma - 1)chi*rho = (gamma - 1)B^2/8/pi.  THIS CURRENTLY DOES NOT WORK PROPERLY (probably too steep for splines)
		smooth - smooth profile before using it for interpolation
		method - either "spline", "pwr" or "brokenpwr"
		bpargs - list of two arrays, each denoting the radius range for fitting each segment of the broken power law; by default, [np.array([1e6,2e8]), np.array([8e8, 1e10])]
		renorm - renormalize fit to central magnetic strength
		checkposderiv - check Bfld derivative to always be positive
		blankfunc - if true, returns a function with Bfld = 0, Bderiv = 0
	"""

	def __init__(self, radius, mass, Bfld, gamma=4./3., filename=False, smooth=False, method="brokenpwr", bpargs=[np.array([1e6,2e8]), np.array([8e8, 1e10])], 
					min_r=1e3, spline_k=3, spline_s=1., renorm=False, checkposderiv=True, blankfunc=False):

		self.gamma = gamma
		self.nabladev = False
		self.spline_k = spline_k
		self.spline_s = spline_s
		self.min_r = min_r

		if blankfunc:
			def fBfld(r, m):
				return 0.*r
			def fBderiv(r, m, drdm):
				return 0.*r
		else:
			#Load in files
			if filename:
				f_in = open(filename, 'r')
				f_in_data = np.loadtxt(f_in)
				self.radius = np.array(f_in_data[:,0])
				self.mass = np.array(f_in_data[:,1])
				self.Bfld = np.array(f_in_data[:,2])
			else:
				self.radius = radius
				self.Bfld = Bfld
				self.mass = mass

			#Smooth magnetic field profile, if necessary
			if smooth:
				self.Bfld = self.movingaverage(self.Bfld, smooth)

			#Derive magnetic field function
			if method == "spline":
				[self.fBfld_r, self.fBderiv_r] = self.getspline_Bfld()
			elif method == "pwr":
				[self.fBfld_r, self.fBderiv_r] = self.getpwr(bpargs, renorm)
			else:
				[self.fBfld_r, self.fBderiv_r] = self.getbrokenpwr(bpargs, renorm)

			if checkposderiv:
				testout = self.fBderiv_r(self.radius)
				args = testout > 0
				if len(testout[args] > 0):
					print "WARNING: positive magnetic slope detected!  Minimum r with positive slope is {0:3e}".format(min(self.radius[args]))

			#Derive mass function
			[self.frass, self.frderiv] = self.getspline_radius()

			[fBfld, fBderiv] = self.getfBfld()

		self.fBfld = fBfld
		self.fBderiv = fBderiv


###################################Function Maker##########################################

	def getfBfld(self):

		#Deepcopy these functions in case they get changed later
		fBfld_r = copy.deepcopy(self.fBfld_r)
		fBderiv_r = copy.deepcopy(self.fBderiv_r)
		frass = copy.deepcopy(self.frass)
		frderiv = copy.deepcopy(self.frderiv)

		max_m = self.mass[-1]

		#Define field and field derivative functions
		def fBfld(r, m):
			if m >= max_m:
				return 0.
			else:
				r_0 = max(frass(m), self.min_r)
				return fBfld_r(r_0)*(r_0/max(r, self.min_r))**2

		def fBderiv(r, m, dmdr):
			if m >= max_m:
				return 0.
			else:
				r_0 = max(frass(m), self.min_r)
				dr0dr = frderiv(m)*dmdr
				return ((r_0/max(r, self.min_r))**2*fBderiv_r(r_0) + 2.*fBfld_r(r_0)*r_0/max(r, self.min_r)**2)*dr0dr - 2*fBfld_r(r_0)*r_0**2/max(r, self.min_r)**3

		return [fBfld, fBderiv]


###################################Fitting Functions#######################################

	#Cubic spline interpolation of magnetic field profile
	def getspline_Bfld(self):
		if self.radius[0]:
			radius_ext = np.concatenate([np.array([0]), self.radius, np.array([1e12])])
			Bfld_ext = np.concatenate([np.array([self.Bfld[0]]), self.Bfld, np.array([1e-3*min(self.Bfld)])])
		else:
			radius_ext = self.radius
			Bfld_ext = self.Bfld
		fBfld = UnivariateSpline(radius_ext, Bfld_ext, k=self.spline_k)
		fBderiv = fBfld.derivative()
		return [fBfld, fBderiv]


	#Cubic spline interpolation of radius as a function of cumulative mass
	def getspline_radius(self):
		if self.radius[0]:
#			radius_ext = np.concatenate([np.array([0]), self.radius, np.array([1e12])])
#			mass_ext = np.concatenate([np.array([0]), self.mass, np.array([max(self.mass)])])
			radius_ext = np.concatenate([np.array([0]), self.radius])
			mass_ext = np.concatenate([np.array([0]), self.mass])
		else:
			radius_ext = self.radius
			mass_ext = self.mass
		frad = UnivariateSpline(mass_ext, radius_ext, k=self.spline_k, s=self.spline_s*len(radius_ext))
		frderiv = frad.derivative()
		if np.isnan(frad(1e33)):
			jesus = lord
		return [frad, frderiv]


	#Fits a broken power law - first the two power law sections are fit to, then the normalization and knee.  The normalization
	#can be scaled afterward ("renorm"), but a single fit to the entire data set with curve_fit favours large values over small.
	def getbrokenpwr(self, bpargs, renorm):
		interior_limits = bpargs[0]
		exterior_limits = bpargs[1]
		args_int = (self.radius > interior_limits[0])*(self.radius < interior_limits[1])
		args_ext = (self.radius > exterior_limits[0])*(self.radius < exterior_limits[1])
		fit_int = np.polyfit(np.log(self.radius[args_int]), np.log(self.Bfld[args_int]), 1)
		fit_ext = np.polyfit(np.log(self.radius[args_ext]), np.log(self.Bfld[args_ext]), 1)
		beta1 = fit_int[0]
		beta2 = fit_ext[0]

		#Yay, I get to use a closure! a = normalization, b = characteristic radius of power law knee
		def func_to_fit(x, a, b):
			return a*((x/b)**(-2*beta1) + (x/b)**(-2*beta2))**-0.5

		fit_output = curve_fit(func_to_fit, self.radius[self.radius < bpargs[1][1]], self.Bfld[self.radius < bpargs[1][1]], p0=[1e11, 1e9])
		B0 = fit_output[0][0]
		r0 = fit_output[0][1]

		if renorm:
			B0 = renorm*((self.radius[0]/r0)**(-2*beta1) + (self.radius[0]/r0)**(-2*beta2))**0.5

		fBfld = lambda x: B0*((x/r0)**(-2*beta1) + (x/r0)**(-2*beta2))**-0.5
		fBderiv = lambda x: (B0/r0)*((x/r0)**(-2*beta1) + (x/r0)**(-2*beta2))**-1.5*(beta1*(x/r0)**(-2*beta1 - 1) + beta2*(x/r0)**(-2*beta2 - 1))

		self.fitresults_Bfld = {"beta1": beta1,"beta2": beta2, "B0": B0, "r0": r0}

		return [fBfld, fBderiv]


	def getpwr(self, bpargs, renorm):
		limits = [bpargs[0][0], bpargs[1][1]]
		args = (self.radius > limits[0])*(self.radius < limits[1])
		fit = np.polyfit(np.log(self.radius[args]), np.log(self.Bfld[args]), 1)

		fBfld = lambda x: np.exp(fit[1])*x**fit[0]
		fBderiv = lambda x: np.exp(fit[1])*fit[0]*x**(fit[0] - 1.)

		self.fitresults_Bfld = {"B0": np.exp(fit[1]),"exponent": fit[0]}

		return [fBfld, fBderiv]


	@staticmethod
	#Same as the moving average in Analyzer.py
	def movingaverage(interval, window_size):
		"""Returns the smoothed curve of interval.  Recently fixed kernel offset and edge effects with ghost cells.

		smoothed = movingaverage(interval, window_size)
			interval: set of data to be smoothed
			window_size: number of elements to smooth over; if an even number is given, +1 is added
		"""

		ghost = int(floor(window_size/2.))
		window_size = 2*ghost + 1
		interval_extend = concatenate([interval[0]*ones(ghost), interval, interval[1]*ones(ghost)])
		return convolve(interval_extend, ones(int(window_size))/float(window_size), 'valid')
