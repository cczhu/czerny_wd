import numpy as np
import pylab
from scipy.interpolate import interp1d
import scipy.integrate as scipyinteg
import os, sys
import myhelm_magstar as myhmag
import rhoTcontours as rtc
from StarModCore import maghydrostar_core
import magprofile_mass as magprof
import Sprofile_mass as sprof


class mhs_steve(maghydrostar_core):
	"""
	Stevenson-based magnetohydrostatic star generator.  Generates spherical 
	WDs with adiabatic temperature profiles using the Helmholtz 
	(http://cococubed.asu.edu/code_pages/eos.shtml) EOS, rigid rotation 
	spherical approximation based on Stevenson 1979.  All values in CGS.

	Parameters (Common across maghydrostar)
	---------------------------------------
	mass : wanted mass (g)
	temp_c : wanted central temperature (K).  If you want to use entropy, 
		then populate S_want (code will then ignore this value and 
		calculate self-consistent temp_c).
	magprofile : magnetic profile object.  Defaults to false, meaning no 
		magnetic field.  To insert a user-defined field, insert a magf object.
		To generate a field with constant delta = B^2/(B^2 + 4pi*Gamma1*Pgas),
		insert a float equal to delta.		
	omega : rigid rotation angular velocity (rad/s).  Defaults to 0 (non-
		rotating).  If < 0, code attempts to estimate break-up omega with 
		self.getomegamax(), if >= 0, uses user defined value.
	L_want : wanted angular momentum.
	mintemp : temperature floor (K), effectively switches from adiabatic 
		to isothermal profile if reached.
	composition : "CO", "Mg" or "He" composition.
	togglecoulomb : includes Coulomb corrections in EOS (corrections are 
		included even if specific entropy becomes negative).
	fakeouterpoint : add additional point to profile where M = mass_want to 
		prevent interpolation attempts from failing.
	stop_invertererr : stop integrating when EOS error is reached.
	stop_mrat : stop integrating when integrated M is larger than 
		stop_mrat*mass.
	stop_positivepgrad : stop integrating when dP/dr becomes positive.
	stop_mindenserr : density floor, below which integration is halted.
		Default is set to 1e-10 to prevent it from ever being reached.  
		Helmholtz sometimes has trouble below this 1e-8; try adjusting this
		value to eliminate inverter errors.
	mass_tol : fractional tolerance between mass wanted and mass produced 
		by self.getstarmodel()
	L_tol : fractional tolerance between L wanted and L produced 
		by self.getrotatingstarmodel()
	omega_crit_tol : when using the self.getomegamax() iterative finder, 
		absolute error tolerance for maximum omega.
	nreps : max number of attempts to find a star in self.getstarmodel().
	stopcount_max : when self.getstarmodel() detects that wanted mass and 
					temp_c/S_want combination may not be achievable, number 
					of stellar integration iterations to take before 
					testing for global extrema.
	verbose : report happenings within code.

	Parameters (Unique to mhs_steve)
	---------------------------------
	S_want : user-specified central entropy (erg/K)
	S_old : entropy profile spline (with respect to mass m) of previous star 
		along runaway track; in cases of extreme convective velocity, switch 
		to using steve_oS derivative functions when calculated entropy profile 
		drops below S_old(m).
	mlt_coeff : ["phil", "wwk", "kippw", "steve"]
		Sets mixing length theory coefficients for calculating velocity 
		and superadiabatic temperature gradients.  "phil" is the standard 
		faire coefficients suggested by Phil Chang; "wwk" is back-derived 
		from Woosley, Wunch and Kuhlen 2004; "kippw" is from 
		Kippenhahn & Wieigert (identical to Cox & Giuli); "steve" 
		is from Stevenson 1979.  Since Stevenson's rotational and magnetic
		corrections to convection are expressed as ratios of velocity and
		temperature gradient, they can be used with any of these mlt_coeffs.
	densest : central density initial estimate for self.getstarmodel().
	omegaest : estimate of rigid rotating angular speed.  Default is False
		- code wil then use 0.75*mystar.L_want/I.
	dontintegrate: don't perform any integration

	Returns
	-------
	mystar : mhs_steve class instance
		If star was integrated and data written, results can be found in
		mystar.data.  Further analysis can be performed with 
		mystar.getenergies,	mystar.getgradients, mystar.getconvection
        and mystar.gettimescales.

	Notes
	-----
	Additional documentation can be found for specific functions within the
	class, and in the parent class maghydrostar_core.  The default behaviour 
    of mhs_steve is to shoot for either a user-specified mass or both a 
    mass and an angular momentum.  It is possible to define an instance of 
    mhs_steve and then use integrate_star to	produce profiles of a 
    known density, central temperature/entropy, and	spin angular velocity 
    Omega.  Be warned, however, that many of the integration parameters, 
    including the value of the temperature floor, the types of 
    superadiabatic temperature gradient deviations used, S_old, and the 
	pressure and density at which to halt integration, are not automatically 
    updated.  MAKE SURE these are set when the class instance is declared!  
    See Examples, below.	

	Examples
	--------
	To build a 1.2 Msun star with solid body rotation Omega = 0.3 s^-1:
	>>> import StarModSteve as SMS
	>>> mystar = SMS.mhs_steve(1.2*1.9891e33, False, False, 
	>>>		omega=0.3, temp_c=5e6, verbose=True)
	The stellar profile can be found under mystar.data, and plotted.  
	For exmaple:
	>>> import matplotlib.pyplot as plt
	>>> plt.plot(mystar.data["R"], mystar.data["rho"], 'r-')
	>>> plt.xlabel("r (cm)")
	>>> plt.ylabel(r"$\rho$ (g/cm$^3$)")
	To generate a series of non-rotating WD profiles with 
	increasing density, we can use maghydrostar's methods:
	>>> import StarModSteve as SMS
	>>> import numpy as np
	>>> import copy as cp
	>>> dens_c = 10.**np.arange(8,9,0.1)
	>>> out_dict = {"dens_c": dens_c,
	>>>     "M": np.zeros(len(dens_c)),
	>>>     "stars": []}
	>>> mystar = SMS.mhs_steve(False, False, False, simd_userot=True, 
	>>>				verbose=True, stop_mrat=False, dontintegrate=True)
	>>> for i in range(len(dens_c)):
	>>>     [Mtot, outerr_code] = mystar.integrate_star(dens_c[i], 0.0, 
	>>>					temp_c=5e6, recordstar=True, outputerr=True)
	>>>     out_dict["M"][i] = Mtot
	>>>     out_dict["stars"].append(cp.deepcopy(mystar.data))
	"""

	def __init__(self, mass, S_want, magprofile=False, omega=0., L_want=0., 
				temp_c=False, mintemp=1e5, composition="CO", togglecoulomb=True,
				S_old=False, mlt_coeff="phil", P_end_ratio=1e-8, ps_eostol=1e-8, 
				fakeouterpoint=False, stop_invertererr=True, 
				stop_mrat=2., stop_positivepgrad=True, stop_mindenserr=1e-10, 
				densest=False, omegaest=False, mass_tol=1e-6, L_tol=1e-6, 
				omega_crit_tol=1e-3, nreps=30, stopcount_max=5, 
				dontintegrate=False, verbose=True):

		# Stop doing whatever if user inserts rotation and B field
		if magprofile and omega != 0.:
			print "You cannot currently insert a magnetic field simultaneously with non-zero rotation!  Quitting."
			return

		maghydrostar_core.__init__(self, mass, temp_c, magprofile=magprofile, 
				omega=omega, L_want=L_want, mintemp=mintemp,
				composition=composition, togglecoulomb=togglecoulomb,
				fakeouterpoint=fakeouterpoint, stop_invertererr=stop_invertererr,
				stop_mrat=stop_mrat, stop_positivepgrad=stop_positivepgrad, 
				stop_mindenserr=stop_mindenserr, mass_tol=mass_tol, L_tol=L_tol, 
				omega_crit_tol=omega_crit_tol, nreps=nreps, 
				stopcount_max=stopcount_max, verbose=verbose)

		self.nablarat_crit = False			# This should only be used for debugging!

		# Initialize nuclear specific energy generation rate
		td = rtc.timescale_data(max_axes=[1e12,1e12])
		self.eps_nuc_interp = td.getinterp2d("eps_nuc")
		if S_old:
			self.populateS_old(S_old)
			if verbose:
				print "S_old defined!  Will use old entropy profile if new one dips below it."
		else:
			self.S_old = False
		self.S_old_reltol = 1e-6	# Relative tolerance to shoot for in connect_S_old

		self.derivatives = self.derivatives_steve
		self.first_deriv = self.first_derivatives_steve
		self.set_mlt_coeff(mlt_coeff)
		if self.verbose:		# self.verbose is implicitly defined in maghydrostar
			print "Stevenson 79 derivative selected!  MLT coefficient = {0:s}".format(mlt_coeff)

		if dontintegrate:
			if self.verbose:
				print "WARNING: integration disabled within mhs_steve!"
		else:
			if self.omega < 0.:
				self.omega = 0.
				self.getmaxomega(P_end_ratio=P_end_ratio, densest=densest, S_want=S_want, ps_eostol=ps_eostol)
			else:
				if L_want:
					self.getrotatingstarmodel_2d(densest=densest, omegaest=omegaest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, damp_nrstep=0.25)
				else:
					self.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol)

			# Checks omega, just to make sure user didn't initialze a "dontintegrate" but set omega < 0
			assert self.omega >= 0.


	def populateS_old(self, S_old):
		"""Records user-defined entropy profile into class.
		"""
		self.S_old = S_old.S_old
		self.dS_old = S_old.dS_old
		self.vconv_Sold = S_old.vconv_Sold


######################################## DERIVATIVES #######################################


	def derivatives_steve(self, y, mass, omega, Fconv, failtrig=[-100], ps_eostol=1e-8, m_step=1e29, isotherm=False, grad_full=False):
		"""
		Derivative that uses Stevenson 1979's formulation for superadiabatic convective deviations.
		"""
		R = y[0]
		press = y[1]
		temp = y[2]

		[dens, entropy] = self.getdens_PT(press, temp, failtrig=failtrig)

		Bfld = self.magf.fBfld(R, mass)
		Pchi = (1./8./np.pi)*Bfld**2

		# Take mag pressure Pchi = 0 for calculating hydro coefficients
		[adgradred, hydrograd, nu, alpha, delta, Gamma1, cP, cPhydro, c_s] = self.geteosgradients(dens, temp, 0.0, failtrig=failtrig)

		dydx = np.zeros(3)
		dydx[0] = 1./(4.*np.pi*R**2.*dens)
		dptotaldm = -self.grav*mass/(4.*np.pi*R**4.) + 1./(6.*np.pi)*omega**2/R
		dydx[1] = dptotaldm 	#- Pchi_grad*dydx[0]

		if isotherm:

			hydrograd = 0.		# Zero out hydrograd and deviation; totalgrad then will equal 0.
			nabla_terms = {"v_conv_st": 0., "c_s_st": c_s, "nd": 0.}	# Populate deviations as zero

		else:

			nabla_terms = {"c_s_st": c_s}

#			agrav_eff = -dptotaldm/dydx[0]/dens		# g_eff = -dP/dr/rho
#			nabla_terms["c_s_st"] = (agrav_eff*H_P)**0.5					# c_s = sqrt(g*H_P) (Stevenson 79 sentence below Eqn. 37)

			agrav = self.grav*mass/R**2.			# g_eff = Gm/r^2
			H_P = min(-press*dydx[0]/dptotaldm, (press/self.grav/dens**2)**0.5)	# H_P = min(-P/(dP/dR), sqrt(P/G\rho^2)) (Eggleton 71 approx.)

			nabla_terms["v_conv_st"] = (delta*agrav*H_P/cP/temp*Fconv/dens)**(1./3.)

			if omega == 0.:
				nabla_terms["nd"] = (1./delta)*(nabla_terms["v_conv_st"]/nabla_terms["c_s_st"])**2
			else:
				nabla_terms["nd"] = (1./delta)*(nabla_terms["v_conv_st"]/nabla_terms["c_s_st"])*(2.*H_P*omega/nabla_terms["c_s_st"])

		if self.nablarat_crit and (abs(nabla_terms["nd"])/hydrograd > self.nablarat_crit):
			raise AssertionError("ERROR: Hit critical nabla!  Code is now designed to throw an error so you can jump to the point of error.")

		totalgrad = hydrograd + nabla_terms["nd"]
		dydx[2] = temp/press*totalgrad*dydx[1]

		if grad_full:
			return [dydx, Bfld, Pchi, hydrograd, totalgrad, nabla_terms]
		else:
			return dydx


	def first_derivatives_steve(self, dens, M, Pc, Tc, omega, failtrig=[-100]):
		"""First step to take for self.derivatives_steve()
		"""

		R = (3.*M/(4.*np.pi*dens))**(1./3.)
		moddens = 4./3.*np.pi*dens
		P = Pc - (3.*self.grav/(8.*np.pi)*moddens**(4./3.) - 0.25/np.pi*omega**2*moddens**(1./3.))*M**(2./3.)	# This is integrated out assuming constant density and magnetic field strength

		Bfld = self.magf.fBfld(R, M)
		Pchi = (1./8./np.pi)*Bfld**2

		[adgradred_dumm, hydrograd, nu_dumm, alpha_dumm, delta, Gamma1, cP, cPhydro_dumm, c_s] = self.geteosgradients(dens, Tc, 0.)	# Central rho, T, and current magnetic pressure since dP/dr = 0 for first step.

		#agrav_eff = -self.grav*M/R**2 + 2./3.*omega**2*R		# -Gm/r^2 + 2\Omega^2 r/3
		#H_P = min(Pc*R/(Pc - P), (Pc/self.grav/dens**2)**0.5)	# H_P = -P/(dP/dR) = -Pc*(R - 0)/(P - Pc) = Pc*R/(Pc - P), or sqrt(P/G\rho^2) (Eggleton 71 approx.)

		nabla_terms = {"v_conv_st": 0., "c_s_st": c_s, "nd": 0.}

		totalgrad = hydrograd + nabla_terms["nd"]	# Curently nabla_terms["nd"] = 0

		temp = Tc + Tc/Pc*totalgrad*(P - Pc)		

		# If we hit the temperature floor, artificially force temp to remain at it, and set integration to isothermal
		if temp <= self.mintemp:
			temp = self.mintemp
			isotherm = True
		else:
			isotherm = False

		dy_est = np.array([R/M, (P - Pc)/M, (temp - Tc)/M])

		return [R, P, temp, Bfld, Pchi, hydrograd, totalgrad, nabla_terms, dy_est, isotherm]


	def set_mlt_coeff(self, mlt_type):
		"""Sets MLT coefficients to some variant in literature.

		Parameters
		----------
		mlt_type : ["phil", "wwk", "kippw", "steve"]
					"phil" is the standard faire coefficients suggested by Phil (though probably only used in PC08)
					"wwk" is back-derived from Woosley, Wunch and Kuhlen 2004
					"kippw" is from Kippenhahn & Wieigert (identical to Cox & Giuli)
					"steve" is from Stevenson 1979
		"""

		if mlt_type == "wwk":
			self.nab_coeff = 1./2.
			self.vc_coeff = (4.)**(1./3.)
		elif mlt_type == "kippw":
			self.nab_coeff = 8.
			self.vc_coeff = (1./4)**(1./3.)
		elif mlt_type == "steve":
			self.nab_coeff = 25.*np.pi**2/6.
			self.vc_coeff = (25./4./np.pi*(2./5.)**2.5)**(1./3.)
		else:
			self.nab_coeff = 1.
			self.vc_coeff = 1.


	def derivatives_oS(self, y, mass, omega, dens_est, temp_est, failtrig=[-100], ps_eostol=1e-8, m_step=1e29, isotherm=False, grad_full=False):
		"""
		Derivative that replaces standard derivative when S(m) < S_old(m)
		"""

		R = y[0]
		press = y[1]

		failtrig_oS = np.zeros(2)	# Account for multiple points of failure in function

		# max(self.S_old(mass), 0.) is a hack to prevent interpolant from going below zero (which occasionally happens due to noisy S curves)
		[dens, temp, gamma_dummy] = self.getdens_PS_est(press, max(self.S_old(mass), 0.), failtrig=failtrig, dens_est=dens_est, temp_est=temp_est, eostol=ps_eostol)
		failtrig_oS[0] = failtrig[0]

		Bfld = self.magf.fBfld(R, mass)
		Pchi = (1./8./np.pi)*Bfld**2

		# Take mag pressure Pchi = 0 for calculating hydro coefficients
		[adgradred, hydrograd, nu, alpha, delta, Gamma1, cP, cPhydro, c_s] = self.geteosgradients(dens, temp, 0.0, failtrig=failtrig)
		failtrig_oS[1] = failtrig[0]

		dydx = np.zeros(2)
		dydx[0] = 1./(4.*np.pi*R**2.*dens)
		dptotaldm = -self.grav*mass/(4.*np.pi*R**4.) + 1./(6.*np.pi)*omega**2/R
		dydx[1] = dptotaldm 	#- Pchi_grad*dydx[0]

		if isotherm:

			hydrograd = 0.		# Zero out hydrograd and deviation; totalgrad then will equal 0.
			nabla_terms = {"v_conv_st": 0., "c_s_st": c_s, "nd": 0.}	# Populate deviations as zero

		else:

			nabla_terms = {"c_s_st": c_s}	# Populate deviations as zero
			nabla_terms["v_conv_st"] = self.vconv_Sold(mass)

			H_P = min(-press*dydx[0]/dptotaldm, (press/self.grav/dens**2)**0.5)	# H_P = min(-P/(dP/dR), sqrt(P/G\rho^2)) (Eggleton 71 approx.)

			nabla_terms["nd"] = -H_P/cP*self.dS_old(mass)/dydx[0]	# ds/dr = ds/dm*dm/dr

		if self.nablarat_crit and (abs(nabla_terms["nd"])/hydrograd > self.nablarat_crit):
			raise AssertionError("ERROR: Hit critical nabla!  Code is now designed to throw an error so you can jump to the point of error.")

		failtrig[0] = max(failtrig_oS)	# Return largest value for failtrig

		if grad_full:
			totalgrad = hydrograd + nabla_terms["nd"]
			dTdm = temp/press*totalgrad*dydx[1]
			return [dydx, dTdm, Bfld, Pchi, hydrograd, totalgrad, nabla_terms]
		else:
			return dydx


######################################## INTEGRATOR ########################################


	def integrate_star(self, dens_c, temp_c, omega, recordstar=False, P_end_ratio=1e-8, ps_eostol=1e-8, outputerr=False):
		"""
		ODE solver that generates stellar profile given a central density dens_c, central 
		temperature temp_c, and solid-body angular speed omega.  Uses Stevenson 1979
		convection model.  All arguments not listed below are same as in __init__().

		Parameters
		----------
		dens_c : central density (g/cm^3)
		temp_c : central temperature (K)
		omega : solid-body angular speed (rad/s)
		outputerr : output any error codes received from integrator
		"""

		# If we use self.nabladev make sure it isn't negative
		if self.nabladev:
			assert self.nabladev > 0.

		stepsize = max(self.mass_want,0.4*self.Msun)*0.01*min(self.mass_tol, 1e-6)	# Found out the hard way that if you use simd_usegammavar, having a large mass_tol can cause errors

		outerr_code = False		# Integrator error code (see loop)

		# Print out the M = 0 step
		M = 0
		R = 0
		temp = temp_c
		dens = dens_c
		[Pc, entropy] = self.getpress_rhoT(dens, temp)		# P IS THE **GAS** PRESSURE, NOT THE TOTAL PRESSURE!
		Pend = Pc*P_end_ratio
		if recordstar:			# Auto-clears any previous data dict
			self.data = {"M": [M],
			"R": [R],
			"Pgas": [Pc],
			"rho": [dens],
			"T": [temp],
			"Sgas": [entropy],
			"Pmag": [],			# Pmag, B will be populated in the next step (done in case B(r = 0) = infty)
			"B": [],
			"nabla_hydro": [],
			"nabla_mhdr": [],
			"nabla_terms": []}	# dy not listed here

		errtrig = [0]		# Errortrig to stop integrator when Helmholtz NR-method fails to properly invert the EOS (usually due to errors slightly beyond tolerance)

		# Load convection luminosity data ("_st" used to distinguish from post-processed convective values)
		self.fconv_data = {"Lnuc_st": [0],
					"Lconv_st": [0],
					"Fconv_st": [0],
					"eps_nuc_st": [self.eps_nuc_interp(dens, temp)]}

		# Take one step forward (we know the central values, and we assume dP_chi/dr = 0), assuming the central density does not change significantly
		# first_deriv also returns the starting value of isotherm
		M = stepsize
		R, P, temp, Bfld, Pmag, hydrograd, totalgrad, nabla_terms, dy, isotherm = self.first_deriv(
																				dens, M, Pc, temp, omega, failtrig=errtrig)
		[dens, entropy] = self.getdens_PT(P, temp, failtrig=errtrig)
		self.advanceFconv(dens, R, temp, stepsize)

		# If the inversion (will only be used if simd_usegammavar is active) in very first step fails, throw up an error and quit
		if errtrig[0]:
			outerr_code = "firststep_inverter_err"
			if outputerr:
				return [M, outerr_code]
			else:
				return M

		if recordstar:
			self.data["M"].append(M)
			self.data["R"].append(R)
			self.data["Pgas"].append(P)
			self.data["rho"].append(dens)
			self.data["T"].append(temp)
			self.data["Sgas"].append(entropy)
			self.data["Pmag"].append(Pmag)			# Pmag, B, etc, for first step calculated
			self.data["B"].append(Bfld)
			self.data["dy"] = dy.reshape((1,3))
			self.data["nabla_hydro"].append(hydrograd)
			self.data["nabla_mhdr"].append(totalgrad)
			self.data["nabla_terms"].append(nabla_terms)

		S_old_toggle = False

		# Continue stepping using scipy.integrate.odeint
		while P > Pend:

			# Adaptive stepsize
			if S_old_toggle:
				y_in = np.array([R, P])
				[dy, dTdm, Bfld, Pmag, hydrograd, totalgrad, nabla_terms] = self.derivatives_oS(y_in, M, omega, dens, temp, ps_eostol=ps_eostol, 
																						m_step=stepsize, isotherm=isotherm, grad_full=True)
				stepsize = self.stepcoeff*min(abs(y_in/(dy+1e-35)))
				dy = np.append(dy, np.array([dTdm]))

			else:
				y_in = np.array([R, P, temp])
				[dy, Bfld, Pmag, hydrograd, totalgrad, nabla_terms] = self.derivatives(y_in, M, omega, self.fconv_data["Fconv_st"][-1],
																		ps_eostol=ps_eostol, m_step=stepsize, isotherm=isotherm, grad_full=True)
				stepsize = self.stepcoeff*min(abs(y_in/(dy+1e-35)))

			if recordstar:
				self.data["Pmag"].append(Pmag)
				self.data["B"].append(Bfld)
				self.data["dy"] = np.append(self.data["dy"], dy.reshape((1,3)), 0)
				self.data["nabla_hydro"].append(hydrograd)
				self.data["nabla_mhdr"].append(totalgrad)
				self.data["nabla_terms"].append(nabla_terms)

			if S_old_toggle:
				R, P = scipyinteg.odeint(self.derivatives_oS, y_in, np.array([M,M+stepsize]), 
														args=(omega, dens, temp, errtrig, ps_eostol, stepsize, isotherm), 
														h0=stepsize*0.01, hmax = stepsize, mxstep=1000)[1,:]
				entropy = self.S_old(M + stepsize)
				[dens, temp, gamma_dummy] = self.getdens_PS_est(P, entropy, failtrig=errtrig, dens_est=dens, temp_est=temp, eostol=ps_eostol)

			else:
				R, P, temp = scipyinteg.odeint(self.derivatives, y_in, np.array([M,M+stepsize]), 
														args=(omega, self.fconv_data["Fconv_st"][-1], errtrig, ps_eostol, stepsize, isotherm), 
														h0=stepsize*0.01, hmax = stepsize, mxstep=1000)[1,:]

				[dens, entropy] = self.getdens_PT(P, temp, failtrig=errtrig)

			if self.S_old and entropy <= self.S_old(M + stepsize) and not S_old_toggle:
				R, P, temp, M = self.connect_S_old(y_in, M, stepsize, omega, Pend, errtrig, ps_eostol)
				[dens, entropy] = self.getdens_PT(P, temp, failtrig=errtrig)
				S_old_toggle = True
			elif temp <= self.mintemp and not isotherm:
				R, P, temp, M = self.connect_isotherm(y_in, M, stepsize, omega, Pend, errtrig, ps_eostol, S_old_toggle, denstempest=[dens, temp])
				[dens, entropy] = self.getdens_PT(P, temp, failtrig=errtrig)
				isotherm = True
			else:
				M += stepsize

			self.advanceFconv(dens, R, temp, stepsize)

			if recordstar:
				self.data["M"].append(M)
				self.data["R"].append(R)
				self.data["Pgas"].append(P)
				self.data["rho"].append(dens)
				self.data["T"].append(temp)
				self.data["Sgas"].append(entropy)

			# Check errors
			if self.stop_invertererr and errtrig[0]:
				outerr_code = "inverter_err"
				print "helmeos or inverter error!  Writing last data-point and stopping integration."
				break
			if P < 0:
				outerr_code = "negativep_err"
				print "P < 0!  Writing last data-point and stopping integration."
				break
			if self.stop_positivepgrad and dy[1] > 0:
				outerr_code = "positivegrad_err"
				print "dP/dR > 0!  Writing last data-point and stopping integration."
				break
			if self.stop_mrat and (M > self.stop_mrat*self.mass_want):
				outerr_code = "hugemass_err"
				print "M is huge!"
				break
			if self.stop_mindenserr and (abs(dens) < self.stop_mindenserr):
				if self.s_mind_errcode:
					outerr_code = "mindens_err"
				print "Density is below {0:e}!  Writing last data-point and stopping integration.".format(self.stop_mindenserr)
				break

		# Step outward one last time
		if S_old_toggle:
			y_in = np.array([R, P])
			[dy, dTdm, Bfld, Pmag, hydrograd, totalgrad, nabla_terms] = self.derivatives_oS(y_in, M, omega, dens, temp, ps_eostol=ps_eostol, 
																					m_step=stepsize, isotherm=isotherm, grad_full=True)
			dy_export = np.append(dy, np.array([dTdm]))
		else:
			y_in = np.array([R, P, temp])
			[dy, Bfld, Pmag, hydrograd, totalgrad, nabla_terms] = self.derivatives(y_in, M, omega, self.fconv_data["Fconv_st"][-1], ps_eostol=ps_eostol, m_step=stepsize, isotherm=isotherm, grad_full=True)
			dy_export = np.array(dy)

		# Generate a fake data point where M is exactly mass_want.  Useful for setting initial conditions for 3D WD simulations.
		if recordstar:
			self.data["Pmag"].append(Pmag)
			self.data["B"].append(Bfld)
			self.data["dy"] = np.append(self.data["dy"], dy_export.reshape((1,3)), 0)
			self.data["nabla_hydro"].append(hydrograd)
			self.data["nabla_mhdr"].append(totalgrad) 
			self.data["nabla_terms"].append(nabla_terms) 

			if self.fakeouterpoint:
				stepsize = self.stepcoeff*min(abs(y_in/(dy+1e-30)))
				if S_old_toggle:
					y_out = scipyinteg.odeint(self.derivatives_oS, y_in, np.array([M,M+stepsize]), 
														args=(omega, dens, temp, errtrig, ps_eostol, stepsize, isotherm), 
														h0=stepsize*0.01, hmax = stepsize, mxstep=1000)[1,:]
				else:
					y_out = scipyinteg.odeint(self.derivatives, y_in, np.array([M,M+stepsize]), args=(omega, self.fconv_data["Fconv_st"][-1], errtrig, ps_eostol, stepsize, isotherm), h0=stepsize*0.01, hmax = stepsize)[1,:]
				self.advanceFconv(0., y_out[0], 0., Mstep)
				self.data["M"].append(max(self.mass_want, M+stepsize))
				self.data["R"].append(y_out[0])
				self.data["Pgas"].append(0.)
				self.data["rho"].append(0.)
				self.data["T"].append(0.)
				self.data["Sgas"].append(0.)
				y_in = y_out	# Gradients one last time
				if S_old_toggle:
					[dy, Bfld, Pmag, hydrograd, totalgrad, nabla_terms] = self.derivatives_oS(y_in, M, omega, ps_eostol=ps_eostol, m_step=stepsize, isotherm=isotherm, grad_full=True)
				else:
					[dy, Bfld, Pmag, hydrograd, totalgrad, nabla_terms] = self.derivatives(y_in, M, omega, self.fconv_data["Fconv_st"][-1], ps_eostol=ps_eostol, m_step=stepsize, isotherm=isotherm, grad_full=True)
				self.data["Pmag"].append(Pmag)
				self.data["B"].append(Bfld)
				self.data["dy"] = np.append(self.data["dy"], dy.reshape((1,3)), 0)
				self.data["nabla_hydro"].append(hydrograd)
				self.data["nabla_mhdr"].append(totalgrad)
				self.data["nabla_terms"].append(nabla_terms)

			self.unpack_nabla_terms()
			self.unpack_Fconv()
			for item in self.data.keys():
				self.data[item] = np.array(self.data[item])

		if outputerr:
			return [M, outerr_code]
		else:
			return M


	def connect_isotherm(self, y_in, M, stepsize, omega, Pend, errtrig, ps_eostol, S_old_toggle, denstempest=[1e6,1e7], iter_max=1000, subtol=1e-8):
		"""
		Stevenson 1979-compatible loop to achieve self.mintemp within tolerance self.mintemp_reltol.
		"""
		substep = 0.1*stepsize			# we'll use constant stepsizes
		ys_in = np.array(y_in)			# arrays are passed by reference, not by value!
		P = ys_in[1]
		sM = M
		i = 0

		if S_old_toggle:
			dens = denstempest[0]
			temp = denstempest[1]

		while i < iter_max and P > Pend and abs(substep) > subtol*stepsize:			# Integrate forward toward self.mintemp

			if S_old_toggle:

				R, P = scipyinteg.odeint(self.derivatives_oS, ys_in, np.array([sM,sM+substep]), 
												args=(omega, dens, temp, errtrig, ps_eostol, substep, False), 
												h0=substep*0.01, hmax = substep, mxstep=1000)[1,:]
				entropy = self.S_old(sM+substep)
				[dens, temp, gamma_dummy] = self.getdens_PS_est(P, entropy, failtrig=errtrig, dens_est=dens, temp_est=temp, eostol=ps_eostol)

			else:
				R, P, temp = scipyinteg.odeint(self.derivatives, ys_in, np.array([sM,sM+substep]), 
												args=(omega, self.fconv_data["Fconv_st"][-1], errtrig, ps_eostol, substep, 
												False), h0=substep*0.01, hmax = substep, mxstep=1000)[1,:]

			sM += substep

			# If we're within the temperature tolerance
			if abs(temp - self.mintemp)/self.mintemp < self.mintemp_reltol:
				break

			# If we've overshot
			if temp < self.mintemp:
				if S_old_toggle:
					R, P = np.array(ys_in)
				else:
					R, P, temp = np.array(ys_in)	# reset R, P, temp to start of step
				sM -= substep					# reverse mass step
				substep = 0.1*substep			# repeat last integration step with greater accuracy

			# If we've done neither, set up next integration
			if S_old_toggle:
				ys_in = np.array([R, P])
			else:
				ys_in = np.array([R, P, temp])
				
		return [R, P, temp, sM]


	def connect_S_old(self, y_in, M, stepsize, omega, Pend, errtrig, ps_eostol, iter_max=1000, subtol=1e-8):
		"""
		Loop to achieve an entropy of self.S_old within tolerance self.S_old_reltol.
		"""
		substep = 0.1*stepsize			# we'll use constant stepsizes
		ys_in = np.array(y_in)			# arrays are passed by reference, not by value!
		P = ys_in[1]
		sM = M
		i = 0
		while i < iter_max and P > Pend and abs(substep) > subtol*stepsize:			# Integrate forward toward self.mintemp

			R, P, temp = scipyinteg.odeint(self.derivatives, ys_in, np.array([sM,sM+substep]), 
											args=(omega, self.fconv_data["Fconv_st"][-1], errtrig, ps_eostol, substep, 
											False), h0=substep*0.01, hmax = substep, mxstep=1000)[1,:]

			[dens, entropy] = self.getdens_PT(P, temp, failtrig=errtrig)

			sM += substep

			# If we're within the temperature tolerance
			S_hit = self.S_old(sM)
			if abs(entropy - S_hit)/S_hit < self.S_old_reltol:
				break

			# If we've overshot
			if entropy < S_hit:
				R, P, temp = np.array(ys_in)	# reset R, P, temp to start of step
				sM -= substep					# reverse mass step
				substep = 0.1*substep			# repeat last integration step with greater accuracy

			# If we've done neither, set up next integration
			ys_in = np.array([R, P, temp])
				
		return [R, P, temp, sM]


	def advanceFconv(self, dens, R, T, Mstep):
		"""
		Integrates Fconv, step by step using trapezoidal rule
		(https://en.wikipedia.org/wiki/Trapezoidal_rule), where 
		F_i+1 = F_i + 0.5*dx_i*(y_i+1 + y_i).  Here, dens and T are 
		used to calculate eps_i+1, and then L_i+1 = L_i + 
		0.5*dM_i*(eps_i + eps_i+1) and F_i+1 = L_i+1/(4*pi*R_i+1). Currently
		NO DISTINCTION is made between L_nuc and L_conv.

		In integrate_star, advanceFconv must be called AFTER a derivative step 
		is called, but BEFORE the stepsize is recalculated.  This gives us the
		proper dens and T to calculate eps_i+1, but keeps dM as dM_i.
		"""
		# Calculate eps_i+1
		self.fconv_data["eps_nuc_st"].append(self.eps_nuc_interp(dens, T))
		# Calculate L_i+1 = L_i + 0.5*dM_i*(eps_i + eps_i+1)
		self.fconv_data["Lnuc_st"].append(self.fconv_data["Lnuc_st"][-1] + 0.5*Mstep*(self.fconv_data["eps_nuc_st"][-1] + self.fconv_data["eps_nuc_st"][-2]))

		# NOT CURRENTLY BEING USED FOR ANYTHING
#		[new_eint, new_edeg, dumm_c_s] = self.gethelmeos_energies(dens, T)
#		self.fconv_data["eth"].append(new_eint - new_edeg)				# Calculate CURRENT eth
#		self.fconv_data["Eth"].append(self.fconv_data["Eth"][-1] + 0.5*Mstep*(self.fconv_data["eth"][-1] + self.fconv_data["eth"][-2]))

		# Right now, HARDCODING Lconv = Lnuc
		self.fconv_data["Lconv_st"].append(self.fconv_data["Lnuc_st"][-1])

		# F_i+1 = L_i+1/(4*pi*R_i+1).
		self.fconv_data["Fconv_st"].append(self.fconv_data["Lconv_st"][-1]/(4.*np.pi*R**2))


	def unpack_nabla_terms(self):
		len_nabla = len(self.data["nabla_terms"])
		for item in self.data["nabla_terms"][0].keys():
			self.data[item] = np.zeros(len_nabla)
		for i in range(len_nabla):
			for item in self.data["nabla_terms"][0].keys():
				self.data[item][i] = self.data["nabla_terms"][i][item]
		del self.data["nabla_terms"]	# delete nabla_terms to remove duplicate copies


	def unpack_Fconv(self):
		for item in self.fconv_data.keys():
			self.data[item] = np.array(self.fconv_data[item])


########################################### POSTPROCESSING FUNCTIONS #############################################


	def getgradients(self):
		"""Obtains magnetohydrostatic gradients for diagnosing stellar structures.  Requires a profile to have already been made (but can be run on partially complete sets).
		"""

		len_arr = len(self.data["rho"])
		self.data["nabla_ad"] = np.zeros(len_arr)
		self.data["nu"] = np.zeros(len_arr)
		self.data["alpha"] = np.zeros(len_arr)
		self.data["delta"] = np.zeros(len_arr)
		self.data["gamma_ad"] = np.zeros(len_arr)
		self.data["cP"] = np.zeros(len_arr)

		#Calculate stepsizes
		m_step = np.zeros(len_arr)
		m_step[:-1] = self.data["M"][1:] - self.data["M"][:-1]
		m_step[-1] = m_step[-2]

		for i in range(len_arr):
			y_in = np.array([self.data["R"][i], self.data["Pgas"][i], self.data["T"][i]])
			[adgradred, hydrograd, self.data["nu"][i], self.data["alpha"][i], self.data["delta"][i], self.data["gamma_ad"][i], self.data["cP"][i], cPydro_dumm, c_s_dumm] = self.geteosgradients(self.data["rho"][i], self.data["T"][i], self.data["Pmag"][i])
			self.data["nabla_ad"][i] = (self.data["Pgas"][i] + self.data["Pmag"][i])/self.data["T"][i]*adgradred
		self.data["dy"][0] = np.array(self.data["dy"][1])		#derivatives using standard function are undefined at R = 0.


	def getconvection_vconv(self):
		"""overwrite StarModCore/getconvection_vconv
		"""
		self.data["vconv"] = self.vc_coeff*(self.data["delta"]*self.data["agrav"]*self.data["H_Preduced"]/self.data["cP"]/self.data["T"]*self.data["Fconv"]/self.data["rho"])**(1./3.)
		self.data["vnuc"] = self.vc_coeff*(self.data["delta"]*self.data["agrav"]*self.data["H_Preduced"]/self.data["cP"]/self.data["T"]*self.data["Fnuc"]/self.data["rho"])**(1./3.)	# Equivalent convective velocity of entire nuclear luminosity carried away by convection


########################################### BLANK STARMOD FOR POST-PROCESSING #############################################


	def backcalculate_subloop(self):
		"""mhs_steve subloop to backcalculate.
		"""

		datalength = len(self.data["M"])
		m_step = self.data["M"][1:] - self.data["M"][:-1]
		m_step = np.concatenate( [m_step, np.array([m_step[-1]])] )

		# Create temporary housing for data
		temp_data = {"Pgas": np.zeros(datalength),
					"Sgas": np.zeros(datalength),
					"Pmag": np.zeros(datalength),
					"nabla_hydro": np.zeros(datalength),
					"nabla_mhdr": np.zeros(datalength),
					"dy": np.zeros([datalength,3])
					}

		self.data["nabla_terms"] = []	# Not very pretty, but a blank class instance shouldn't have initialized these in the first place.
		self.fconv_data = {"Lnuc_st": [0],
					"Lconv_st": [0],
					"Fconv_st": [0],
					"eps_nuc_st": [self.eps_nuc_interp(self.data["rho"][0], self.data["T"][0])]}

		for i in range(datalength):
			[temp_data["Pgas"][i], temp_data["Sgas"][i]] = self.getpress_rhoT(self.data["rho"][i], self.data["T"][i])
			if i > 0:
				self.advanceFconv(self.data["rho"][i], self.data["R"][i], self.data["T"][i], m_step[i-1])
			y_in = np.array([self.data["R"][i], temp_data["Pgas"][i], self.data["T"][i]])
			[temp_data["dy"][i], dummyBfld, temp_data["Pmag"][i], temp_data["nabla_hydro"][i], temp_data["nabla_mhdr"][i], nabla_terms] = self.derivatives(y_in, self.data["M"][i], self.omega, self.fconv_data["Fconv_st"][i], m_step=m_step[i], grad_full=True)
			self.data["nabla_terms"].append(nabla_terms)

		# Because derivatives_steve wasn't meant to be used at the origin
		temp_data["nabla_mhdr"][0] = temp_data["nabla_mhdr"][1]

		self.unpack_nabla_terms()
		self.unpack_Fconv()

		return temp_data
