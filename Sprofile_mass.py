import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import os, sys
import copy

################## ENTROPY_PROFILE CLASS ###############################

class entropy_profile:
	"""
	Entropy/convective velocity profiler.  Returns functions S_old, dS_old, and
	vconv_Sold, used in StarModSteve.mhs_steve for end-of-simmering 
	calculations.

	Parameters
	----------
	mass : mass
	Sgas : S(m)
	vconv : v_conv(m)
	filename : read profile from file (class constructor will not read radius 
		or Bfld in that case)
	smooth : smooth profile before using it for interpolation
	spline_k : UnivariateSpline degree of smoothing spline (k must be <=5)
	mass_cut : mass cutoff for input entropy and mass curves used to produce
		interpolated S_old function.  UnivariateSpline uses a derivative-based
		outer bundary Sometimes necessary due to the extreme
		derivatives of Sgas close to the surface of the star.
	blankfunc : if true, returns a function with S_old = 0, dS_old = 0, 
			vconv_Sold = 0
	"""

	def __init__(self, mass, Sgas, vconv, filename=False, smooth=False,
					spline_k=3, mass_cut=0.99, blankfunc=False):

		self.spline_k = spline_k
		self.mass_cut = mass_cut
#		self.spline_s = spline_s

		if blankfunc:
			def S_old(m):
				return 0.*m
			def dS_old(m):
				return 0.*m
			def vconv_Sold(m):
				return 0.*m
		else:
			# Load in files
			if filename:
				f_in = open(filename, 'r')
				f_in_data = np.loadtxt(f_in)
				self.mass = np.array(f_in_data[:,0])
				self.Sgas = np.array(f_in_data[:,1])
				self.vconv = np.array(f_in_data[:,2])
			else:
				self.Sgas = Sgas
				self.vconv = vconv
				self.mass = mass

			# Smooth profiles, if necessary
			if smooth:
				self.Sgas = self.movingaverage(self.Sgas, smooth)
				self.vconv = self.movingaverage(self.vconv, smooth)

			# Derive functions
			[S_old, dS_old, vconv_Sold] = self.getspline_Sold()

		self.S_old = S_old
		self.dS_old = dS_old
		self.vconv_Sold = vconv_Sold


	def getspline_Sold(self):
		"""Cubic spline interpolation of entropy and convective velocity.
		"""
		want = self.mass < max(self.mass)*self.mass_cut
		S_old = UnivariateSpline(self.mass[want], self.Sgas[want], k=self.spline_k)
		dS_old = S_old.derivative()
		vconv_Sold = UnivariateSpline(self.mass[want], self.vconv[want], k=self.spline_k)
		return [S_old, dS_old, vconv_Sold] 


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
