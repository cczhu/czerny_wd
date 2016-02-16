import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import os, sys
import copy

################## MAGPROFILE CLASS ###############################

class magprofile:
	"""
	Magnetic field profiler.  Returns function fBfld used in
	StarMod.maghydrostar to determine and propagate magnetic field.

	Parameters
	----------
	radius : radius
	mass : mass(r) profile
	Bfld : B(r) profile
	rho : rho(r) profile
	filename : read profile from file (class constructor will not read radius 
		or Bfld in that case)
	smooth : smooth profile before using it for interpolation
	min_r : minimum radius for spline interpolant; below this, use rho_est/rhoc;
		defaults to -1, which sets min_r to the minimum radius when rho/rhoc < 0.99
	spline_k : UnivariateSpline degree of smoothing spline (k must be <=5)
	spline_s : UnivariateSpline positive smoothing factor used to choose 
		number of knots
	checkposderiv : check Bfld derivative to always be positive
	blankfunc : if true, returns a function with Bfld = 0, Bderiv = 0
	"""

	def __init__(self, radius, mass, Bfld, rho, filename=False, 
					smooth=False, min_r=-1, spline_k=3, spline_s=1., 
					checkposderiv=True, blankfunc=False):

		self.spline_k = spline_k
		self.spline_s = spline_s

		if blankfunc:

			def fBfld(r, m):
				return 0.*r

		else:

			if min_r >= 0:
				self.min_r = min_r
			else:
				self.min_r = radius[min(np.where(rho < 0.99*max(rho))[0])]
			self.rhoc = rho[0]

			# Load in files
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

			# Smooth magnetic field profile, if necessary
			if smooth:
				self.Bfld = self.movingaverage(self.Bfld, smooth)

			# Derive magnetic field function
			self.fBfld_r = self.getspline_Bfld()

			if checkposderiv:
				testout = self.fBfld_r(self.radius)
				args = np.where((testout[1:] - testout[:-1])/(testout[1:] + testout[:-1]) > 5e-4)[0]
				if len(args) > 0:
					print "WARNING: positive magnetic slope detected!  Minimum r with positive slope is {0:3e}".format(min(self.radius[args]))

			# Derive mass function
			self.frass = self.getspline_radius()

			fBfld = self.getfBfld()

		self.fBfld = fBfld

###################################Function Maker##########################################

	def getfBfld(self):

		# Deepcopy these functions in case they get changed later
		fBfld_r = copy.deepcopy(self.fBfld_r)
		frass = copy.deepcopy(self.frass)

		max_m = self.mass[-1]

		# Define field and field derivative functions
		def fBfld(r, m):
			if m >= max_m:
				return 0.
			r_0 = frass(m)
			if r_0 < self.min_r:
				rho_n = 3.*m/(4.*np.pi*max(r, 1e-30)**3)
				return fBfld_r(r_0)*(rho_n/self.rhoc)**(2./3.)
			else:
				return fBfld_r(r_0)*(r_0/r)**2

		return fBfld


################################### Fitting Functions #######################################

	def getspline_Bfld(self):
		"""Cubic spline interpolation of magnetic field profile.
		"""

		if self.radius[0]:
			radius_ext = np.concatenate([np.array([0]), self.radius, np.array([1e12])])
			Bfld_ext = np.concatenate([np.array([self.Bfld[0]]), self.Bfld, np.array([1e-3*min(self.Bfld)])])
		else:
			radius_ext = self.radius
			Bfld_ext = self.Bfld
		fBfld = UnivariateSpline(radius_ext, Bfld_ext, k=self.spline_k, ext=3)

		return fBfld


	def getspline_radius(self):
		"""Cubic spline interpolation of radius as a function of cumulative mass.
		"""

		if self.radius[0]:
			radius_ext = np.concatenate([np.array([0]), self.radius])
			mass_ext = np.concatenate([np.array([0]), self.mass])
		else:
			radius_ext = self.radius
			mass_ext = self.mass
		frad = UnivariateSpline(mass_ext, radius_ext, k=self.spline_k, s=self.spline_s*len(radius_ext), ext=3)
		if np.isnan(frad(1e33)):
			raise AssertionError("Univariate spline instance unable to take a massive radius!")
		return frad

	@staticmethod
	def movingaverage(interval, window_size):
		"""Returns the smoothed curve of interval.  Recently fixed kernel offset and edge effects with ghost cells.

		Arguments:
		interval: set of data to be smoothed
		window_size: number of elements to smooth over; if an even number is given, +1 is added
		"""

		ghost = int(floor(window_size/2.))
		window_size = 2*ghost + 1
		interval_extend = concatenate([interval[0]*ones(ghost), interval, interval[1]*ones(ghost)])
		return convolve(interval_extend, ones(int(window_size))/float(window_size), 'valid')
