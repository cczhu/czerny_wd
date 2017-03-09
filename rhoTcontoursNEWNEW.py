import numpy as np
import myhelm_magstar as myhmag
import pylab
import scipy.interpolate as interp

class timescale_data:
	"""Class for holding and performing basic calculations on timescale data.  Allows user to add custom timescale lines.

	timescales = timescale_data(max_axes=[10**10.,10**10.]):
		max_axes: density and temperature upper limits for 2D arrays
	"""

	def __init__(self, max_axes=[10**10.,10**10.]):

		self.sectoyear = 60.0**2*24.0*365.0
		self.grav = 6.67384e-8

		mf = open("/home/cczhu/martenwdmaker/geteosetc/co_new.out",'r-')
		data = np.loadtxt(mf, skiprows = 3)

		self.data = {}

		# T, Rho, P, E, S, Cp, eta, gamma1, gamma3, eps_neu, eps_nuc
		self.data["T"] = data[:,0]
		self.data["rho"] = data[:,1]
		self.data["P"] = data[:,2]
		self.data["E"] = data[:,3]
		self.data["S"] = data[:,4]
		self.data["Cp"] = data[:,5]	
		self.data["eta"] = data[:,6]		#I have no idea what this is
		self.data["gamma1"] = data[:,7]
		self.data["gamma3"] = data[:,8]
		self.data["eps_neu"] = data[:,9]	#neutrino losses
		self.data["eps_nuc"] = data[:,10]	#carbon fusion

		self.data["tau_neu"] = self.data["Cp"]*self.data["T"]/self.data["eps_neu"]/self.sectoyear
		self.data["tau_nuc"] = self.data["Cp"]*self.data["T"]/self.data["eps_nuc"]/self.sectoyear	#tau_cc, or t_h

		self.data["tau_cdyn"] = (self.grav*self.data["rho"])**(-0.5)/self.sectoyear
		self.data["tau_dyn"] = 7.36*(self.grav*self.data["rho"])**(-0.5)/self.sectoyear # 1/sqrt(\bar{rho}) = 7.36/sqrt(rho_c), from K&W, Table 19.1 (5/3 polytrope)
		self.getP0()

		self.data["tau_eq_neunuc"] = self.data["tau_neu"] - self.data["tau_nuc"]
		self.data["tau_eq_cdnuc"] = self.data["tau_cdyn"] - self.data["tau_nuc"]
		self.data["tau_eq_dynnuc"] = self.data["tau_dyn"] - self.data["tau_nuc"]
		self.data["Prat"] = self.data["P"]/self.data["P0"]

		#Raw, unreduced density/temperature
		self.rho_r = np.unique(self.data["rho"])
		self.T_r = np.unique(self.data["T"])
		self.len_r = np.array([len(self.rho_r), len(self.T_r)])

		#Make reduced 2D array for plotting
		self.getdatap(max_axes)


	def getP0(self):
		"""Calculates degeneracy component of the pressure."""
		myhmag.initializehelmholtz()
		abar = 13.714285714285715
		zbar = abar/2.0
		self.data["P0"] = np.zeros(len(self.data["rho"]))
		for i in range(len(self.data["rho"])):
			self.data["P0"][i],energ,sound,gammaout,entropy,dummyfail = myhmag.gethelmholtzeos(1000.,self.data["rho"][i],abar,zbar,True)


	def getdatap(self, max_axes):
		"""Obtain reduced window "_p" density and temperature and data values for plotting - useful if you want to auto-print clabels or reduce the number of points
		   being plotted.  You need to use this function to obtain 2D arrays for plotting regardless; just set the limits really high.
		"""

		self.rho_p = self.rho_r[self.rho_r < max_axes[0]]
		self.T_p = self.T_r[self.T_r < max_axes[1]]
		self.len_p = np.array([len(self.rho_p), len(self.T_p)])

		self.data_p = {}	#Start with a fresh copy of the data

		for item in self.data.keys():
			self.data_p[item] = self.make2Darray(self.rho_p, self.T_p, self.data[item], self.len_r[0])

	def getinterp2d(self, key_want):
		if not self.data_p.has_key(key_want):	#Just in case
			self.data_p[item] = self.make2Darray(self.rho_p, self.T_p, self.data[key_want], self.len_r[0])
		i2d = interp.interp2d(self.rho_p, self.T_p, self.data_p[key_want], bounds_error=True)
		def my_i2d(rho, T):
			"""2D interpolator created by timescale_data class.

				value = my_i2d(rho, T)
			"""
			rho_a = min(max(self.rho_p), max(min(self.rho_p), rho))
			T_a = min(max(self.T_p), max(min(self.T_p), T))
			return i2d(rho_a, T_a)[0]
		return my_i2d


	def get_taudyn_line(self):

		T_zero = np.zeros(self.len_p[0])
		for i in range(self.len_p[0]):
			T_zero[i] = self.findzero(self.T_p,self.data_p["tau_eq_dynnuc"][:,i])
		f = interp.interp1d(self.rho_p, T_zero)
		return f


	def get_tauneunuc_line(self):

		T_zero = np.zeros(self.len_p[0])
		for i in range(self.len_p[0]):
			T_zero[i] = self.findzero(self.T_p,self.data_p["tau_eq_neunuc"][:,i])
		f = interp.interp1d(self.rho_p, T_zero)
		return f


	@staticmethod
	def make2Darray(x,y,z,xlenfull):
		xlen = len(x)
		ylen = len(y)
		new = np.zeros((ylen,xlen))
		for i in range(ylen):
			new[i,:] = z[i*xlenfull:i*xlenfull + xlen]  #i.e. new[y[i],x[i]] = z something; since in contour y is row # and x is col #
		return new


	@staticmethod
	def findzero(x,y):
		"""Finds root for a monatonic sampled function y(x).
		"""
		yc = np.array(y)
		yc[np.isinf(y)] = -1e100
		args = np.where(yc > 0)[0]
		if len(args) == 0:
			print "Unable to find zero! Using maximum T."
			i = len(yc) - 1
		else:
			i = min(args)

		#Do a fit using the surrounding points, and perform interpolation based on that
		p = np.polyfit(x[i-1:i+1],yc[i-1:i+1],1)
		yfit = p[0]*x[i-1:i+1] + p[1]
		f = interp.interp1d(yfit, x[i-1:i+1])
		return float(f(0))


def get_timescale_lines(ax, td):
	"""Plots timescales - useful for generic plotting, but to make specific figures you'll probably want to write your own function."""
	
	neur_N = [1e6, 1e4, 1e2, 1e0, 1e-2]
	neur_clabel = [r'$\tau_{\nu}=10^6$',r'$10^4$',r'$10^2$',r'$10^0$',r'$10^{-2}$']

	for i in range(len(neur_N)):
		cs = ax.contour(td.rho_p, td.T_p, td.data_p["tau_neu"], [neur_N[i]], colors='b', linestyles='dotted')
		ax.clabel(cs, fontsize=14, inline=1, fmt=neur_clabel[i])	#inline_spacing = -50

	nucr_N = [1e6, 1e4, 1e2, 1e0, 1e-2,1e-4,1e-6]
	nucr_clabel = [r'$\tau_\mathrm{cc}=10^6$',r'$10^4$',r'$10^2$',r'$10^0$',r'$10^{-2}$',r'$10^{-4}$',r'$10^{-6}$']

	for i in range(len(nucr_N)):
		cs = ax.contour(td.rho_p, td.T_p, td.data_p["tau_nuc"], [nucr_N[i]], colors='r', linestyles=':')
		ax.clabel(cs, fontsize=14, inline=1, fmt=nucr_clabel[i])

	S_N = [1e8, 10**8.1, 10**8.2]
	S_clabel = [r'$S = 10^8$',r'$10^{8.1}$',r'$10^{8.2}$']

	for i in range(len(S_N)):
		cs = ax.contour(td.rho_p, td.T_p, td.data_p["S"], [S_N[i]], colors='g', linestyles='dotted', linewidth=4)
		ax.clabel(cs, fontsize=14, inline=1, fmt=S_clabel[i])

	cs = ax.contour(td.rho_p, td.T_p, td.data_p["tau_eq_neunuc"], [0], colors='m', linestyles='-', linewidth=6)
	ax.clabel(cs, fontsize=14, inline=1, fmt=r'$\tau_\mathrm{cc}=\tau_{\nu}$')

	cs = ax.contour(td.rho_p, td.T_p, td.data_p["Prat"], [2.0], colors='k', linestyles='--', linewidth=6)
	ax.clabel(cs, fontsize=14, inline=1, fmt=r'$P = 2P(T=0)$')

	cs = ax.contour(td.rho_p, td.T_p, td.data_p["tau_eq_cdnuc"], [0], colors='r', linestyles=':', linewidth=4)
	ax.clabel(cs, fontsize=14, inline=1, fmt=r'$\tau_\mathrm{cc}=(G\rho_c)^{-1/2}$')

	cs = ax.contour(td.rho_p, td.T_p, td.data_p["tau_eq_dynnuc"], [0], colors="r", linestyles='-', linewidths=2)
	ax.clabel(cs, fontsize=14, inline=1, fmt=r"$\tau_\mathrm{cc} = \tau_\mathrm{dyn}$")



def get_figure(width=9., height=7.5, axes=[10**5.5,10**8.5,10**7.8,10**9.75], timescales=True):
	pylab.rc('font', family="serif")
	pylab.rc('font', size=14)
	fig = pylab.figure(figsize=(width,height))
	ax = fig.add_axes([1.0/width, 0.75/height, (width - 1.2)/width, (height - 1.)/height])

	ax.axis(axes)
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_xlabel(r"$\rho$ (g cm$^{-3}$)", fontsize=16)
	ax.set_ylabel(r"$T$ (K)", fontsize=16)

	if timescales:
		td = timescale_data(max_axes=[axes[1], axes[3]])
		get_timescale_lines(ax, td)

	return [fig, ax]
