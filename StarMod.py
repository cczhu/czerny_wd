import numpy as np
import pylab
from scipy.interpolate import interp1d
import scipy.integrate as scipyinteg
import os, sys
import magprofile_mass as magprof
import myhelm_magstar as myhmag
import rhoTcontours as rtc
import copy

################HYDROSTAR CLASS###############################

class maghydrostar:
	"""
	Magnetohydrostatic star generator.  Generates spherical WDs with 
	adiabatic temperature profiles using the Helmholtz 
	(http://cococubed.asu.edu/code_pages/eos.shtml) EOS, rigid rotation 
	spherical approximation and Gough & Tayler 66 magnetic convective 
	suppression criterion (but no pressure term).  Includes gravitational 
	and varying specific heat ratio terms of GT66 criterion.  All values 
	in CGS.

	Parameters
	----------
	mass : wanted mass (g)
	temp_c : wanted central temperature (K).  If you want to use entropy, 
		then populate S_want (code will then ignore this value and 
		calculate self-consistent temp_c).
	magprofile : magnetic profile object.  Defaults to false, meaning no 
		magnetic field.  If derivtype="sim", a magf object containing the 
		field should be passed.  If derivtype="simcd", magprofile should 
		equal delta = B^2/(B^2 + 4pi*Gamma1*Pgas).
	omega : rigid rotation angular velocity (rad/s).  Defaults to 0 (non-
		rotating).  If < 0, attempts to estimate break-up omega with 
		self.getomegamax(), if >= 0, uses user defined value.
	Lwant : wanted angular momentum.
	S_want : use a user-specified central entropy (erg/K) INSTEAD OF 
		temperature temp_c.
	mintemp : temperature floor (K), effectively switches from adiabatic 
		to isothermal profile if reached.
	composition : "CO", "Mg" or "He" composition.
	derivtype : derivative function used - either "sim" (default) or "simcd" 
		(assumes constant magnetic delta = B^2/(B^2 + 4pi*Gamma1*Pgas)).
	simd_userot : use Solberg-Hoiland deviation from adiabatic temperature 
		gradient.
	simd_usegammavar : use gamma = c_P/c_V index magnetic deviation from 
		adiabatic temperature gradient.
	simd_usegrav : use gravity magnetic devation from adiabatic temperature 
		gradient.
	simd_suppress : suppresses deviations from adiabaticity in first step of 
		integration.
	nablarat_crit : software crash toggle if nabla ever becomes too 
		large - only useful for developmental purposes!
	P_end_ratio : ratio of P/P_c at which to terminate stellar integration.
	ps_eostol : tolerance ONLY IN myhmag.geteosinversionsp_withest RIGHT NOW 
		(when inverting Helmholtz EOS to find density/temperature from 
		pressure/entropy).
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
	densest : central density initial estimate for self.getstarmodel().
	omegaest : estimate of rigid rotating angular speed.  Default is False
		- code wil then use 0.75*mystar.L_want/I.
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
	dontintegrate : set up the problem, but don't shoot for a star.
	verbose : report happenings within code.

	Returns
	-------
	mystar : maghydrostar class instance
		If star was integrated and data written, results can be found in
		mystar.data.  Further analysis can be performed with 
		mystar.getenergies,	mystar.getgradients and mystar.getconvection.


	Notes
	-----
	Additional documentation can be found for specific functions within the
	class.  The default behaviour of maghydrostar is to shoot for either a 
	user-specified mass or both a mass and an angular momentum.  It is possible
	to define an instance of maghydrostar and then use integrate_star to
	produce profiles of a known density, central temperature/entropy, and
	spin angular velocity Omega.  Be warned, however, that many of the 
	integration parameters, including the value of the temperature floor, the
	types of superadiabatic temperature gradient deviations used, and the 
	pressure and density at which to halt integration.  MAKE SURE these are set 
	when the class instance is declared!  See Examples, below.		


	Examples
	--------
	To build a 1.2 Msun star with solid body rotation Omega = 0.3 s^-1:
	>>> import StarMod as Star
	>>> mystar = Star.maghydrostar(1.2*1.9891e33, 5e6, False, 
	>>>		omega=0.3, simd_userot=True, verbose=True)
	The stellar profile can be found under mystar.data, and plotted.  
	For exmaple:
	>>> import matplotlib.pyplot as plt
	>>> plt.plot(mystar.data["R"], mystar.data["rho"], 'r-')
	>>> plt.xlabel("r (cm)")
	>>> plt.ylabel(r"$\rho$ (g/cm$^3$)")
	To generate a series of adiabatic non-rotating WD profiles with 
	increasing density, we can use maghydrostar's methods:
	>>> import StarMod as Star
	>>> import numpy as np
	>>> import copy
	>>> dens_c = 10.**np.arange(8,9,0.1)
	>>> out_dict = {"dens_c": dens_c,
	>>>     "M": np.zeros(len(dens_c)),
	>>>     "stars": []}
	>>> mystar = Star.maghydrostar(False, False, False, simd_userot=True, 
	>>>				verbose=True, stop_mrat=False, dontintegrate=True)
	>>> for i in range(len(dens_c)):
	>>>     [Mtot, outerr_code] = mystar.integrate_star(dens_c[i], 5e6, 0.0, 
	>>>					recordstar=True, outputerr=True)
	>>>     out_dict["M"][i] = Mtot
	>>>     out_dict["stars"].append(copy.deepcopy(mystar.data))
	"""

	def __init__(self, mass, temp_c, magprofile=False, omega=0., Lwant=0., 
				S_want=False, mintemp=1e5, composition="CO", derivtype="sim",
				simd_userot=True, simd_usegammavar=False, simd_usegrav=False, 
				simd_suppress=False, nablarat_crit=False, P_end_ratio=1e-8, 
				ps_eostol=1e-8, fakeouterpoint=False, stop_invertererr=True, 
				stop_mrat=2., stop_positivepgrad=True, stop_mindenserr=1e-10, 
				densest=False, omegaest=False, mass_tol=1e-6, L_tol=1e-6, 
				omega_crit_tol=1e-3, nreps=30, stopcount_max=5, 
				dontintegrate=False, verbose=True):

		self.Msun = 1.9891e33
		self.grav = 6.67384e-8
		self.stepcoeff = 1e-2
		self.mass_want = mass
		self.mass_tol = mass_tol
		self.temp_c = temp_c
		self.nreps = nreps
		self.derivtype = derivtype
		self.nablarat_crit = nablarat_crit

		# recent additions to __init__ (formerly were just passed to getstarmodel).
		self.fakeouterpoint = fakeouterpoint
		self.stop_invertererr = stop_invertererr
		self.stop_mrat = stop_mrat
		self.stop_positivepgrad = stop_positivepgrad
		self.stop_mindenserr = stop_mindenserr
		self.s_mind_errcode = False	# If True, returns mindenserr as an outerr_code (usually unwanted behaviour)
		self.stopcount_max = stopcount_max
		self.omega_crit_tol = omega_crit_tol
		self.L_want = Lwant
		self.L_tol = L_tol

		# Remember to print messages
		self.verbose = verbose

		# If magprofile is false, generate a zero-field profile
		if not magprofile:
			if self.verbose:
				print "magprofile == False - will assume star has no magnetic field!"
			self.magf = magprof.magprofile(None, None, None, blankfunc=True)
		else:
			self.magf = magprofile

		self.mintemp = mintemp
		self.mintemp_reltol = 1e-6
		if self.verbose:
			print "Minimum temperature set to {0:.3e}".format(self.mintemp)

		self.simd_userot = simd_userot
		self.simd_usegammavar = simd_usegammavar
		self.simd_usegrav = simd_usegrav
		self.simd_suppress = simd_suppress

		if self.simd_usegammavar and self.verbose:
			print "VARIABLE GAMMA TERM included in nabla deviation!"
		if self.simd_usegrav and self.verbose:
			print "GRAVITY TERM included in nabla deviation!"
		if self.simd_userot and self.verbose:
			print "ROTATION TERM included in nabla deviation!"
		if self.simd_suppress and self.verbose:
			print "WARNING - YOU'LL BE SUPPRESSING nabla DEVIATIONS IN THE FIRST MASS STEP!  I HOPE YOU HAVE A GOOD REASON!"

		if omega < 0.:
			if self.verbose:
				print "Omega < 0 - max rotation estimator selected!"
			self.omega = 0.
		else:
			if self.verbose:
				print "Omega = {0:.3e}".format(omega)
			self.omega = max(0., omega)

		if composition == "He":
			if self.verbose:
				print "Composition set to HELIUM"
			self.abar = 4.0
		elif composition == "Mg":
			if self.verbose:
				print "Composition set to MAGNESIUM"
			self.abar = 24.0
		else:
			if self.verbose:
				print "Composition set to CARBON-OXYGEN (50% mix by mass)"
			self.abar = 13.714285714285715
		self.zbar = self.abar/2.	# Will need changing when we start looking at weirder stars.

		# In the future, adapt other EOSs?

		# Run a dummy gethelmeos to check if EOS is initialized:
		pressure,energy,soundspeed,gammaout,entropy,checkeosfailure = myhmag.gethelmholtzeos(1e5,1e2,2.,4.,True)
		if checkeosfailure:					# If 1 is returned...
			print "I noticed you haven't initialized helmholtz.  Doing so now."
			myhmag.initializehelmholtz()	# ...initialize helmholtz

		if self.derivtype == "simcd":
			if magprofile:
				self.nabladev = magprofile
				if type(self.nabladev) != float:
					raise AssertionError("ERROR: if you use simcd derivtype, then magprofile must be (a float) equal to delta(r) = B^2/(B^2 + 4pi*Gamma1*Pgas)")
			else:
				self.nabladev = 0.
			if self.verbose:
				print "GT66/MM09 derivative WITH CONSTANT DEVIATION RATIO selected! Using self.nabladev as deviation delta(r) (MM09 Eqn. 4)."
		else:
			if self.verbose:
				print "GT66/MM09 derivative selected!"
		# derivatives_gtsh now handles both constant deviation and user-defined magnetic profile.
		self.derivatives = self.derivatives_gtsh
		self.first_deriv = self.first_derivatives_gtsh

		# Set temperature floor - essentially necessary since Helmholtz has a 1000 K temperature floor.
#		self.mintemp_func = self.mintemp_func_creator()		#lambda x: 1./(np.exp(-100.*(x - 1.125*self.mintemp)/self.mintemp) + 1.)	#**2

		self.data = {}
		if dontintegrate:
			if self.verbose:
				print "WARNING: integration disabled within maghydrostar!"
		else:
			if omega < 0.:
				self.getmaxomega(P_end_ratio=P_end_ratio, densest=densest, S_want=S_want, ps_eostol=ps_eostol)
			else:
				if Lwant:
					self.getrotatingstarmodel_2d(densest=densest, omegaest=omegaest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, damp_nrstep=0.25)
				else:
					self.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol)


######################################### SHOOTING ALGORITHMS #########################################

	def getstarmodel(self, densest=False, omega_user=-1, S_want=False, P_end_ratio=1e-8, ps_eostol=1e-8, deltastepcoeff=0.1, out_search=False):
		"""
		Newton-Raphson solver to obtain WD with user-specified mass.  Arguments not listed 
		below have identical meanings as class initialization ones.

		Parameters
		----------
		omega_user : use omega_user instead of self.omega in integrate_star.  
			Defaults to -1 (meaning "DO NOT USE") - code then uses self.omega.
		deltastepcoeff : when estimating the Jacobian, Delta rho = 
			deltastepcoeff*abs(deltadens_previous).  Defaults to 0.1.
		out_search : prints the rho and corresponding M values calculated by 
			integrate_star during the shooting process
		"""

		if not densest:
			densest = 3.*3.73330253e-60*self.mass_want**2	# The 3. is recently added to reduce the time for integrating massive WDs from scratch
		deltadens = densest

		# If we want a specific central entropy rather than temperature
		if S_want:
			[pressure_dummy, self.temp_c] = self.getpress_rhoS(densest, S_want)

		# Use either stored omega or user-override omega (useful for calculating dL/dOmega in self.getjacobian_omega())
		if omega_user < 0:
			omega = self.omega
		else:
			omega = omega_user

		i = 0
		[Mtot, outerr_code] = self.integrate_star(densest, self.temp_c, omega, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, outputerr=True)	# First shot
		if self.verbose:
			print "First shot: M = {0:.6e} (vs. M_want = {1:.6e}; relative err. {2:.6e})".format(Mtot, self.mass_want, abs(Mtot - self.mass_want)/self.mass_want)

		# If we incur hugemass_err the first time (occurs for very high entropy stars)
		if outerr_code == "hugemass_err":
			if self.verbose:
				print "hugemass_err is huge from the first estimate.  Let's try a much lower density."
			densest = 0.1*densest
			[Mtot, outerr_code] = self.integrate_star(densest, self.temp_c, omega, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, outputerr=True)
			if outerr_code:			# If we incur any error (or hugemass_err again)
				print "OUTERR_CODE {0:s}!  EXITING FUNCTION!".format(outerr_code)
				return outerr_code

		# Keep past shooting attempts
		M_arr = np.array([Mtot])
		dens_arr = np.array([densest])
		checkglobalextreme = False

		while abs(Mtot - self.mass_want) >= self.mass_tol*self.mass_want and i < self.nreps and not outerr_code:

			[dMdrho, outerr_code] = self.getjacobian_dens(densest, self.temp_c, omega, Mtot, deltastepcoeff*abs(deltadens), 
													S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol)	# Calculate Jacobian
			deltadens = (self.mass_want - Mtot)/dMdrho		# Calculate rho_new = rho_old - delta M/dMdrho
			densest += deltadens

			if densest <= 1e1:	# Central density estimate really shouldn't be lower than about 1e4 g/cc
				densest = abs(deltadens)*0.1
			if self.verbose:
				print "dMdrho is {0:.6e}, deltadens is {1:.6e}".format(dMdrho, deltadens)
				print "Old rho = {0:.6e}; new rho = {1:.6e}".format(densest - deltadens, densest)

#			if abs(deltadens/densest) < 2e-6:
#				print "deltadens/densest is approaching floating point precision - ending integration!"
#				outerr_code = "tinystepmod_err"
#				continue

			if S_want:			# This automatically sets self.temp_c to be consistent with S_want
				[pressure_dummy, self.temp_c] = self.getpress_rhoS(densest, S_want)

			[Mtot, outerr_code] = self.integrate_star(densest, self.temp_c, omega, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, outputerr=True)
			if self.verbose:
				print "Current shot: M = {0:.6e} (vs. M_want = {1:.6e}; relative err. {2:.6e})".format(Mtot, self.mass_want, abs(Mtot - self.mass_want)/self.mass_want)
			M_arr = np.append(M_arr, Mtot)
			dens_arr = np.append(dens_arr, densest)

			# Check for extreme circumstances where a solution might be impossible
			if int((dens_arr[i] - dens_arr[i-1])/abs(dens_arr[i] - dens_arr[i-1])) != int((M_arr[i] - M_arr[i-1])/abs(M_arr[i] - M_arr[i-1])) and not checkglobalextreme:
				print "Density and mass don't have a positive-definite relationship in the regime you selected!"
				if i == 1:	# Check that increasing (decreasing) central density increases (decreases) mass
					print "We're at the beginning of mass finding (i = 1), so reversing direction!"
					stepmod *= -1.0
				checkglobalextreme = True
				stopcount = 0

			# If we've already activated checkglobalextreme, we should only integrate up to stopcount = stopcount_max
			if checkglobalextreme:
				stopcount += 1
				if stopcount > self.stopcount_max:		#If we've integrated one times too many after checkglobalextreme = True
					outerr_code = self.chk_global_extrema(M_arr, Mtot - self.mass_want, self.mass_want)

			i += 1

		if i == self.nreps:
			print "ERROR, maximum number of shooting attempts {0:d} reached!".format(i)
			return "nrepsnotconverged_err"

		# If we incur any error
		if outerr_code:
			print "OUTERR_CODE {0:s}!  EXITING FUNCTION!".format(outerr_code)
			return outerr_code

		# This piece is here (and not in getrotatingstarmodel()) because we don't have recordstar=True in the loop (but we need it for rotating stars)
		if self.verbose:
			print "Final shot!"
		Mtot = self.integrate_star(densest, self.temp_c, omega, recordstar=True, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol)

		if abs((Mtot - self.mass_want)/self.mass_want) > self.mass_tol:
			print "ERROR! (M_total - mass_want)/mass_want = {0:.6e}  THIS IS BIGGER THAN YOUR TOLERANCE!  CHECK YOUR ICS!".format((Mtot - self.mass_want)/self.mass_want)
		else:
			print "(M_total - mass_want)/mass_want = {0:.6e}".format((Mtot - self.mass_want)/self.mass_want)

		if out_search:
			return [M_arr, dens_arr]


	def getjacobian_dens(self, densest, temp_c, omega, Mtot, drho, S_want=False, P_end_ratio=1e-8, ps_eostol=1e-8):
		"""Returns dM/drho for dMdrho*drho = M - Mwant
		"""
		if S_want:
			[pressure_dummy, temp_c] = self.getpress_rhoS(densest + drho, S_want)
		[Mtot_drho, outerr_code] = self.integrate_star(densest + drho, temp_c, omega, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, outputerr=True)
		return [(Mtot_drho - Mtot)/drho, outerr_code]


	def getrotatingstarmodel(self, densest=False, omegaest=False, S_want=False, P_end_ratio=1e-8, ps_eostol=1e-8, damp_nrstep=1, deltastepcoeff=0.05, 
									interior_dscoeff=0.1, omega_warn=10., out_search=False):
		"""
		Newton-Raphson solver to obtain rigidly-rotating WD with user-specified mass and 
		ANGULAR MOMENTUM.  If you wish to create a WD with user-specified mass and omega, 
		use self.getstarmodel().  Arguments not defined below have identical meanings as 
		class initialization ones.

		Parameters
		----------
		damp_nrstep : damping term for Newton-Raphson stepping.  Defaults to 1, 
			but that will likely lead to overshooting near critical Omega values.
		deltastepcoeff: when estimating the Jacobian, Delta Omega = 
			deltastepcoeff*abs(deltaOmega_previous).  Defaults to 0.05.
		interior_dscoeff: deltastepcoeff for getstarmodel mass-finding sub-loops.  
			Defaults to 0.1.
		omega_warn: stop integration if self.omega approaches omega_warn*omega_crit 
			estimate.  Defaults to 10 to prevent premature stoppage.
		out_search: prints L, omega calculated by integrate_star during the 
			shooting process.
		"""

		if self.L_want <= 0.:
			print "Please define self.L_want before using this function!  Quitting."
			return

		# If user doesn't specify, randomly pick a relatively safe omega
		if omegaest:
			self.omega = omegaest
		else:
			self.omega = 0.2

		if self.verbose:
			print "==== GENERATING NEW ROTATING STAR MODEL ===="

		i = 0
		outerr_code = self.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, deltastepcoeff=interior_dscoeff)	# First shot
		Ltot = self.getmomentofinertia(self.data["R"], self.data["rho"])[-1]*self.omega

		if self.verbose:
			print "Omega overloop first shot: L = {0:.6e} (vs. L_want = {1:.6e}; relative err. {2:.6e}) ".format( \
															Ltot, self.L_want, abs(Ltot - self.L_want)/self.L_want)

		delta_omega = self.omega		# Note to self - this shallow-copies self.omega, which is a float (and not some weird class instance thing)

		# Keep past shooting attempts
		L_arr = np.array([Ltot])
		omega_arr = np.array([self.omega])

		while abs(Ltot - self.L_want) >= self.L_tol*self.L_want and i < self.nreps and not outerr_code:

			if self.verbose:
				print "== RUNNING GETSTARMODEL WITHIN GETJACOBIAN_OMEGA =="

			[dLdomega, outerr_code] = self.getjacobian_omega(self.data["rho"][0], Ltot, deltastepcoeff*delta_omega, S_want=S_want,
													P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, interior_dscoeff=interior_dscoeff)		# Calculate Jacobian (self.omega auto-passed)
			delta_omega = damp_nrstep*(self.L_want - Ltot)/dLdomega
			self.omega += delta_omega

			if self.omega < 0.:		#  Negative omega is not defined
				self.omega = abs(delta_omega)*0.1
			if self.verbose:
				print "dLdomega is {0}, delta_omega is {1}".format(dLdomega, delta_omega)
				print "Old omega = {0:.6e}; new omega = {1:.6e}".format(self.omega - delta_omega, self.omega)
				print "== RUNNING GETSTARMODEL WITH REVISED OMEGA =="

			self.getstarmodel(densest=self.data["rho"][0], S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, deltastepcoeff=interior_dscoeff)
			Ltot = self.getmomentofinertia(self.data["R"], self.data["rho"])[-1]*self.omega
			if self.verbose:
				print "Omega overloop shot: L = {0:.6e} (vs. L_want = {1:.6e}; relative err. {2:.6e}) ".format(Ltot, 
																self.L_want, abs(Ltot - self.L_want)/self.L_want)
			L_arr = np.append(L_arr, Ltot); omega_arr = np.append(omega_arr, self.omega)

			if self.omega > omega_warn*self.getcritrot(max(self.data["M"]), self.data["R"][-1]):
				outerr_code = "Omega/Omega_crit > {0:.1e}".format(omega_warn)

			i += 1

		if i == self.nreps:
			print "WARNING, maximum number of shooting attempts {0:d} reached!".format(i)

		# If we incur any error
		if outerr_code:
			print "OUTERR_CODE {0:s}!  EXITING FUNCTION!".format(outerr_code)
			return outerr_code

		if (abs(Ltot - self.L_want)/self.L_want > self.L_tol):
			print "ERROR! (L_tot - L_want)/L_want = {0:.6e}   THIS IS BIGGER THAN YOUR TOLERANCE!  CHECK YOUR ICS!".format( \
																(Ltot - self.L_want)/self.L_want)
		else:
			print "(L_tot - L_want)/L_want = {0:.6e}".format((Ltot - self.L_want)/self.L_want)

		if out_search:
			return [L_arr, omega_arr]


	def getjacobian_omega(self, densest, Ltot, domega, S_want=False, P_end_ratio=1e-8, ps_eostol=1e-8, interior_dscoeff=0.1):
		"""Returns dL/domega for getrotatingstarmodel()
		"""
		omega_alt = self.omega + domega
		outerr_code = self.getstarmodel(densest=densest, omega_user=omega_alt, S_want=S_want, 
										P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, deltastepcoeff=interior_dscoeff)
		Ltot_domega = self.getmomentofinertia(self.data["R"], self.data["rho"])[-1]*omega_alt
		return [(Ltot_domega - Ltot)/domega, outerr_code]

############################# UNCOMMENT IF YOU WANT TO USE 2D JACOBIAN (I HAVEN'T FOUND IT TO BE FASTER #################################

	def getrotatingstarmodel_2d(self, densest=False, omegaest=False, S_want=False, P_end_ratio=1e-8, ps_eostol=1e-8, damp_nrstep=1, deltastepcoeff=0.05, 
									omega_warn=10., out_search=False):
		"""Experimental 2D Newton-Raphson solver to obtain rigidly-rotating WD 
		with user-specified mass and ANGULAR MOMENTUM.  If you wish to create a 
		WD with user-specified mass and omega, use self.getstarmodel().  
		Arguments not defined below have identical meanings as class 
		initialization ones.

		Parameters
		----------
		omegaest : estimate of rigid rotating angular speed; by default, 
			uses 0.75*self.L_want/I
		deltastepcoeff : when estimating the Jacobian, Delta 
			rho = deltastepcoeff*abs(deltadens_previous).  Defaults to 0.1.
		omega_warn : stop integration if self.omega approaches 
			omega_warn*omega_crit estimate
		out_search : prints shot_arr = {M_c, dens_c, L, omega} calculated 
			by integrate_star during the shooting process
		"""

		if self.L_want <= 0. or omegaest < 0.:
			print "DO NOT USE THIS FUNCTION IF OMEGA/ANGULAR MOMENTUM LWANT IS SUPPOSED TO BE ZERO!  THE SOLVER WON'T FORCE THE FINAL SOLUTION TO MAINTAIN OMEGA = 0."
			return

		if not densest:
			densest = 3.*3.73330253e-60*self.mass_want**2	# The 3. is recently added to reduce the time for integrating massive WDs from scratch

		# If we want a specific central entropy rather than temperature
		if S_want:
			[pressure_dummy, self.temp_c] = self.getpress_rhoS(densest, S_want)

		# If user doesn't specify, randomly pick a relatively safe omega
		if omegaest:
			self.omega = omegaest
		else:
			self.omega = 0.2

		i = 0
		[Mtot, outerr_code] = self.integrate_star(densest, self.temp_c, self.omega, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, outputerr=True, recordstar=True)	# First shot
		Ltot = self.getmomentofinertia(self.data["R"], self.data["rho"])[-1]*self.omega

		# If we incur hugemass_err the first time
		if outerr_code == "hugemass_err":
			if self.verbose:
				print "hugemass_err is huge from the first estimate.  Let's try a much lower density."
			densest = 0.1*densest
			[Mtot, outerr_code] = self.integrate_star(densest, self.temp_c, self.omega, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, outputerr=True, recordstar=True)
			if outerr_code:
				print "OUTERR_CODE {0:s}!  EXITING FUNCTION!".format(outerr_code)	# If we incur any error (or hugemass_err again)
				return outerr_code

		if self.verbose:
			print "First shot: M = {0:.6e} (vs. M_want = {1:.6e}; relative err. {2:.6e}), L = {3:.6e} (vs. L_want = {4:.6e}; relative err. {5:.6e}) ".format( \
															Mtot, self.mass_want, abs(Mtot - self.mass_want)/self.mass_want, Ltot, 
															self.L_want, abs(Ltot - self.L_want)/self.L_want)

		deltavals = np.array([densest, self.omega])		# Note to self - this shallow-copies self.omega, since self.omega is a float (and not some weird class instance thing)

		# Keep past shooting attempts
		shot_arr = {"M": np.array([Mtot]),
			"dens": np.array([densest]),
			"L": np.array([Ltot]),
			"omega": np.array([self.omega])
			}

		while (abs(Mtot - self.mass_want) >= self.mass_tol*self.mass_want or \
				abs(Ltot - self.L_want) >= self.L_tol*self.L_want) and i < self.nreps and not outerr_code:

			[J_out, outerr_code] = self.getjacobian_2d(densest, self.temp_c, self.omega, Mtot, Ltot, deltastepcoeff*abs(deltavals), 
													S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol)	# Calculate Jacobian
			F_neg = np.array([self.mass_want - Mtot, self.L_want - Ltot])
			deltavals = damp_nrstep*np.linalg.solve(J_out, F_neg)						# http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.solve.html
			if not np.allclose(np.dot(J_out, deltavals), F_neg):
				outerr_code = "Jacobian_err"
			densest += deltavals[0]
			self.omega += deltavals[1]

			if densest <= 1e1:		# Central density estimate really shouldn't be lower than about 1e4 g/cc
				densest = abs(deltavals[0])*0.1
			if self.omega < 0.:		# In case omega < 0
				self.omega = abs(deltavals[1])*0.1
			if self.verbose:
				print "Jacobian is {0}, deltavals is {1}".format(J_out, deltavals)
				print "Old rho = {0:.6e}; new rho = {1:.6e}".format(densest - deltavals[0], densest)
				print "Old omega = {0:.6e}; new omega = {1:.6e}".format(self.omega - deltavals[1], self.omega)

			if S_want:			# IMPORTANT - if S_want is used in getjacobian_rotating, we need to run this anyway to reset self.temp_c!
				[pressure_dummy, self.temp_c] = self.getpress_rhoS(densest, S_want)

			[Mtot, outerr_code] = self.integrate_star(densest, self.temp_c, self.omega, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, outputerr=True, recordstar=True)
			Ltot = self.getmomentofinertia(self.data["R"], self.data["rho"])[-1]*self.omega
			if self.verbose:
				print "Current shot: M = {0:.6e} (vs. M_want = {1:.6e}; relative err. {2:.6e}), L = {3:.6e} (vs. L_want = {4:.6e}; relative err. {5:.6e}) ".format( \
																Mtot, self.mass_want, abs(Mtot - self.mass_want)/self.mass_want, Ltot, 
																self.L_want, abs(Ltot - self.L_want)/self.L_want)

			shot_arr["M"] = np.append(shot_arr["M"], Mtot); shot_arr["dens"] = np.append(shot_arr["dens"], densest)
			shot_arr["L"] = np.append(shot_arr["L"], Ltot); shot_arr["omega"] = np.append(shot_arr["omega"], self.omega)

			if self.omega > omega_warn*self.getcritrot(max(self.data["M"]), self.data["R"][-1]):
				outerr_code = "Omega/Omega_crit > {0:.1e}".format(omega_warn)

			i += 1

		if i == self.nreps:
			print "WARNING, maximum number of shooting attempts {0:d} reached!".format(i)

		# If we incur any error
		if outerr_code:
			print "OUTERR_CODE {0:s}!  EXITING FUNCTION!".format(outerr_code)
			return outerr_code

		if (abs((Mtot - self.mass_want)/self.mass_want) > self.mass_tol) or (abs(Ltot - self.L_want)/self.L_want > self.L_tol):
			print "ERROR! (M_tot - mass_want)/mass_want = {0:.6e}; (L_tot - L_want)/L_want = {1:.6e}   THIS IS BIGGER THAN YOUR TOLERANCE!  CHECK YOUR ICS!".format( \
																(Mtot - self.mass_want)/self.mass_want, (Ltot - self.L_want)/self.L_want)
		else:
			print "(L_total - L_want)/L_want = {0:.6e}".format((Ltot - self.L_want)/self.L_want)

		if out_search:
			return [M_arr, dens_arr]


	def getjacobian_2d(self, densest, temp_c, omega, Mtot, Ltot, dvals, S_want=False, P_end_ratio=1e-8, ps_eostol=1e-8):
		"""Returns Jacobian matrix J_ij for J*[drho, domega] = [dM, dL]
		"""
		# Step density by drho
		if S_want:
			[pressure_dummy, temp_c_alt] = self.getpress_rhoS(densest + dvals[0], S_want)
		else:
			temp_c_alt = temp_c
		[Mtot_drho, outerr_code] = self.integrate_star(densest + dvals[0], temp_c_alt, omega, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, outputerr=True, recordstar=True)
		Ltot_drho = self.getmomentofinertia(self.data["R"], self.data["rho"])[-1]*omega

		# Step omega by domega (return density to original densest)
		omega_alt = omega + dvals[1]		# Step up omega by domega
		[Mtot_domega, outerr_code] = self.integrate_star(densest, temp_c, omega_alt, P_end_ratio=P_end_ratio, outputerr=True, ps_eostol=ps_eostol, recordstar=True)
		Ltot_domega = self.getmomentofinertia(self.data["R"], self.data["rho"])[-1]*omega_alt

		# Obtain Jacobian
		J_line1 = [(Mtot_drho - Mtot)/dvals[0], (Mtot_domega - Mtot)/dvals[1]]
		J_line2 = [(Ltot_drho - Ltot)/dvals[0], (Ltot_domega - Ltot)/dvals[1]]
		return [np.array([J_line1, J_line2]), outerr_code]

############################# UNCOMMENT IF YOU WANT TO USE 2D JACOBIAN (I HAVEN'T FOUND IT TO BE FASTER #################################


	def getmaxomega(self, densest=False, S_want=False, P_end_ratio=1e-8, ps_eostol=1e-8, out_search=False):
		"""
		Iterative solver to obtain WD with specified mass and largest possible Omega value that does not feature a density inversion due to too much rotational support.  Arguments have identical meanings as class initialization ones.
		"""

		# This code can't run if these checks aren't in place!
		if not self.stop_invertererr:
			print "getmaxomega REQUIRES self.stop_invertererr be true!  Setting it so."
			self.stop_invertererr = True
		if not self.stop_mrat:
			print "getmaxomega REQUIRES self.stop_mrat be used!  Setting it to 2."
			self.stop_mrat = 2.
		if not self.stop_positivepgrad:
			print "getmaxomega REQUIRES self.stop_positivepgrad be true!  Setting it so."
			self.stop_positivepgrad = True

		# First, get a stationary model
		if self.verbose:
			print "Obtaining stationary model."
		self.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol)
		if self.verbose:
			print "Obtained stationary solution; using radius to estimate critical omega value"

		self.omega = 0.30*self.getcritrot(self.data["M"][-2], self.data["R"][-2])	#First, get 30% the critical omega estimate (this is an overestimate, but we want to keep away from errors for now
		omegastep = 0.1

		gsm_output = self.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol)
		if gsm_output:
			if self.verbose:
				print "Initial omega estimate too high, leading to error {0}; reducing omega by half!".format(gsm_output)
			self.omega = 0.5*self.omega
			gsm_output = self.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol)
			assert gsm_output == None
		print "First Omega estimate = {0:.6e}; dOmega = {1:.6e}; error message is {2:s}".format(self.omega, omegastep, gsm_output)
		self.omega += omegastep

		i = 0
		while abs(omegastep/self.omega) >= self.omega_crit_tol and i < self.nreps:
			gsm_output = self.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol)
			if self.verbose:
				print "Omega estimate = {0:.6e}; dOmega = {1:.6e}; error = {2:s}".format(self.omega, omegastep, gsm_output)
			if gsm_output:	# If we overstepped the critical rotation point
				self.omega -= 0.9*omegastep
				omegastep = omegastep*0.1
				if self.verbose:
					print "Reached integration error {0:s}; stepping back and rerunning with finer stepsize dOmega = {1:.6e}.".format(gsm_output, omegastep)
			else:
				self.omega += omegastep	# Otherwise keep adding

		if self.verbose:
			print "Critical Omega {0:.6e} determined to accuracy of {1:.6e}; error {2}".format(self.omega, 10.*omegastep, gsm_output)
			
		if i == self.nreps:
			print "WARNING, maximum number of shooting attempts {0:d} reached!".format(i)


	@staticmethod
	def chk_global_extrema(M_arr, dens_arr, Mwant):
		"""Checks the central density to mass relationship to make sure the mass and temp_c/S_want combination can be reached.
		"""

		Mdiff = M_arr[-1] - Mwant
		if Mdiff < 0 and max(M_arr) < Mwant:	# If the current star is too light, and we've never made a heavy enough star
			# If the global maximum isn't the most recent integration, and we're currently going toward lighter stars
			if (np.argmax(M_arr) - len(M_arr) < 1) and M_arr[-1] < M_arr[-2]:
				return "Global maximum {0:.4e} mass is still too small to satisfy Mwant {1:.4e}".format(max(M_arr), Mwant)
		elif Mdiff > 0 and min(M_arr) > Mwant:	# If the current star is too heavy, and we've never made a light enough star
			if (np.argmin(M_arr) - len(M_arr) < 1) and M_arr[-1] > M_arr[-2]:
			# If the global minimum isn't the most recent integration, and we're currently going toward heavier stars
				return "Global minimum {0:.4e} mass is still too large to satisfy Mwant {1:.4e}".format(min(M_arr), Mwant)
		else:
			return None


	def getcritrot(self, M, R):
		"""Returns critical Omega given M and R.
		"""
		return np.sqrt(M*self.grav/R**3)


	def mintemp_func_creator(self):
		"""Creates mintemp function.
		"""
		def mintemp_func(temp):
			if temp <= self.mintemp:
				return 0.
			elif temp <= 33.*self.mintemp:
				return 1./(np.exp(-50.*(temp - 1.250*self.mintemp)/self.mintemp) + 1.)
			return 1.
		return mintemp_func


###################################### EOS STUFF #######################################

	# EOS functions are passed failtrig as a list (since lists are mutable).
	def getpress_rhoT(self, dens, temp, failtrig=[-100], togglecoulomb=True):
		pressure,energy,soundspeed,gammaout,entropy,failtrig[0] = myhmag.gethelmholtzeos(temp,dens,self.abar,self.zbar,togglecoulomb)
		return [pressure, entropy]


	def geteosgradients(self, dens, temp, Pchi, failtrig=[-100], togglecoulomb=True):
		adgradred,hydrograd,nu,alpha,delta,Gamma1,cP,cPhydro,c_s,failtrig[0] = myhmag.gethelmgrads(temp,dens,Pchi,self.abar,self.zbar,togglecoulomb)
		return [adgradred, hydrograd, nu, alpha, delta, Gamma1, cP, cPhydro, c_s]


	def getpress_rhoS(self, dens, entropy, failtrig=[-100], togglecoulomb=True):
		temp,pressure,energy,soundspeed,gammaout,failtrig[0] = myhmag.geteosinversionsd(dens,self.abar,self.zbar,entropy,togglecoulomb)
		return [pressure, temp]

	def getdens_PT(self, press, temp, failtrig=[-100], togglecoulomb=True):
		dens,energy,soundspeed,gammaout,entropy,failtrig[0] = myhmag.geteosinversionpt(temp,self.abar,self.zbar,press, togglecoulomb)
		return [dens, entropy]


	def getdens_PS(self, press, entropy, failtrig=[-100], togglecoulomb=True):
		temp,dens,energy,soundspeed,gammaout,failtrig[0] = myhmag.geteosinversionsp(self.abar,self.zbar,press,entropy,True,togglecoulomb)
		return [dens, temp]


	def getdens_PS_est(self, press, entropy, failtrig=[-100], dens_est=1e6, temp_est=1e7, togglecoulomb=True, eostol=1e-8):
		"""Identical to getdens_PS, but uses user-defined estimates to help with the Newton-Raphson algorithm.
		"""
		temp,dens,energy,soundspeed,gammaout,failtrig[0] = myhmag.geteosinversionsp_withest(self.abar,self.zbar,dens_est,temp_est,press,entropy,True,togglecoulomb, eostol)
		return [dens, temp, gammaout]


	def getgamma_PD(self, dens, press, failtrig=[-100], togglecoulomb=True):
		"""Obtains Gamma_1 = dP/drho_ad estimate for first_derivatives functions.
		"""
		temp_dummy,energy_dummy,soundspeed_dummy,Gamma1_est,entropy_dummy,failtrig[0] = myhmag.geteosinversionpd(dens,self.abar,self.zbar,press,togglecoulomb)
		return Gamma1_est


	def gethelmeos_energies(self, dens, temp, failtrig=[-100], togglecoulomb=True):
		"""Obtains Gamma_1 = dP/drho_ad estimate for first_derivatives functions.
		"""
		press_dumm,e_int,c_s,gamma_dumm,ent_dumm,dummyfail1 = myhmag.gethelmholtzeos(temp,dens,self.abar,self.zbar,togglecoulomb)
		press_dumm,e_deg,sound_dumm,gamma_dumm,ent_dumm,dummyfail2 = myhmag.gethelmholtzeos(1000.,dens,self.abar,self.zbar,togglecoulomb)
		if dummyfail1 or dummyfail2:
			failtrig[0] = 1
		return [e_int, e_deg, c_s]


######################################## DERIVATIVES #######################################


	def derivatives_gtsh(self, y, mass, omega, failtrig=[-100], ps_eostol=1e-8, m_step=1e29, isotherm=False, grad_full=False):
		"""
		Derivative that uses the "simple" Gough & Tayler 1966 formulation for magnetic suppression of convection in addition to the Solberg-Hoiland criterion for convective stability in a rotating medium.  
		"""
		R = y[0]
		press = y[1]
		temp = y[2]

		[dens, entropy] = self.getdens_PT(press, temp, failtrig=failtrig)

		# Take mag pressure Pchi = 0 for calculating hydro coefficients
		[adgradred, hydrograd, nu, alpha, delta, Gamma1, cP, cPhydro, c_s] = self.geteosgradients(dens, temp, 0., failtrig=failtrig)

		dydx = np.zeros(3)
		dydx[0] = 1./(4.*np.pi*R**2.*dens)
		dydx[1] = -self.grav*mass/(4.*np.pi*R**4.) + 1./(6.*np.pi)*omega**2/R	# -Pchi_grad*dydx[0]

		# Obtain magnetic field
		if self.derivtype == "simcd":
			Bfld = np.sqrt(4.*np.pi*Gamma1*press*self.nabladev/(1. - self.nabladev))
		else:
			Bfld = self.magf.fBfld(R, mass)
		Pchi = (1./8./np.pi)*Bfld**2

#		#insurance policy in case -d\rho/dT diverges
#		mintemp_func_current = self.mintemp_func(temp)
#		Bfld = mintemp_func_current*Bfld

		nabla_terms = {}

		if isotherm:

			hydrograd = 0.		# Zero out hydrograd and deviation; totalgrad then will equal 0.
			deviation = 0.

			if self.simd_usegammavar:	# Populate deviations as zero
				nabla_terms["dlngamdlnP"] = 0.
				nabla_terms["nd_gamma"] = 0.
			if self.simd_usegrav:
				nabla_terms["dlngdlnP"] = 0.
				nabla_terms["nd_grav"] = 0.
			if self.simd_userot:
				nabla_terms["nd_rot"] = 0.

		else:

			# GT66 magnetic deviation
			deviation = 1./delta*Bfld**2/(Bfld**2 + 4.*np.pi*Gamma1*press)

			# Magnetic gamma and gravitational terms from GT 66 Eqn. 3.19.
			if self.simd_usegammavar:
				press_est = press + dydx[1]*m_step
				[dens_dummy, temp_dummy, Gamma1_est] = self.getdens_PS_est(press_est, entropy, failtrig=failtrig, dens_est=dens, temp_est=temp, eostol=ps_eostol)	# assumes adiabatic
				nabla_terms["dlngamdlnP"] = (press/Gamma1)*(Gamma1_est - Gamma1)/(press_est - press)	# dlngamma/dlnP = (P/gamma)*(dgamma/dP)
				nabla_terms["nd_gamma"] = 1./delta*Bfld**2/(Bfld**2 + 4.*np.pi*Gamma1*press)*nabla_terms["dlngamdlnP"]
				deviation += nabla_terms["nd_gamma"]
			if self.simd_usegrav:
				GH_Poverg = min(press*R**4/(dens*self.grav*mass**2), 1./dens)		# GH_P/g
				nabla_terms["dlngdlnP"] = -GH_Poverg*(4.*np.pi*dens - 2.*mass/R**3)							# Second term technically becomes NaN, but this function will never see R = 0 or mass = 0
				nabla_terms["nd_grav"] = 1./delta*Bfld**2/(4.*np.pi*Gamma1*press)*nabla_terms["dlngdlnP"]
				deviation -= nabla_terms["nd_grav"]

			# rotational term
			if self.simd_userot:
				H_Poverg = min(press*R**4/(dens*self.grav**2*mass**2), 1./(self.grav*dens))		# H_P/g
				nabla_terms["nd_rot"] = 4.*(2./3.)**0.5*H_Poverg*omega**2/delta
				deviation += nabla_terms["nd_rot"]

		if self.nablarat_crit and (abs(deviation)/hydrograd > self.nablarat_crit):
			raise AssertionError("ERROR: Hit critical nabla!  Code is now designed to throw an error so you can jump to the point of error.")

		totalgrad = hydrograd + deviation
		dydx[2] = temp/press*totalgrad*dydx[1]

		if grad_full:
			return [dydx, Bfld, Pchi, hydrograd, totalgrad, nabla_terms]
		else:
			return dydx


	def first_derivatives_gtsh(self, dens, M, Pc, Tc, omega, failtrig=[-100]):
		"""First step to take for self.derivatives_gtsh()
		"""

		R = (3.*M/(4.*np.pi*dens))**(1./3.)
		moddens = 4./3.*np.pi*dens
		P = Pc - (3.*self.grav/(8.*np.pi)*moddens**(4./3.) - 0.25/np.pi*omega**2*moddens**(1./3.))*M**(2./3.)	# This is integrated out assuming constant density and magnetic field strength

		[adgradred_dumm, hydrograd, nu_dumm, alpha_dumm, delta, Gamma1, cP_dumm, cPhydro_dumm, c_s_dumm] = self.geteosgradients(dens, Tc, 0.)	# Central rho, T, and current magnetic pressure since dP/dr = 0 for first step.  Now uses Pchi = 0.

		# Obtain magnetic field
		if self.derivtype == "simcd":
			Bfld = np.sqrt(4.*np.pi*Gamma1*Pc*self.nabladev/(1. - self.nabladev))
		else:
			Bfld = self.magf.fBfld(R, M)
		Pchi = (1./8./np.pi)*Bfld**2

		# GT66 Magnetic deviation
		deviation = 1./delta*Bfld**2/(Bfld**2 + 4.*np.pi*Gamma1*P)

		nabla_terms = {}
		# magnetic field gamma and gravitational terms
		if self.simd_usegammavar:
			if Tc > self.mintemp and not self.simd_suppress:
				Gamma1_est = self.getgamma_PD(dens, P, failtrig=failtrig, togglecoulomb=True)
				nabla_terms["dlngamdlnP"] = (Pc/Gamma1)*(Gamma1_est - Gamma1)/(P - Pc)		# dlngamma/dlnP = (P/gamma)*(dgamma/dP)
				nabla_terms["nd_gamma"] = 1./delta*Bfld**2/(Bfld**2 + 4.*np.pi*Gamma1*Pc)*nabla_terms["dlngamdlnP"]
				deviation += nabla_terms["nd_gamma"]
			else:
				nabla_terms["dlngamdlnP"] = 0.
				nabla_terms["nd_gamma"] = 0.
		if self.simd_usegrav:
			if Tc > self.mintemp and not self.simd_suppress:
				nabla_terms["dlngdlnP"] = -4.*np.pi/3.	# Comes from assuming H_P/g = 1/(G*rho)
				nabla_terms["nd_grav"] = 1./delta*Bfld**2/(4.*np.pi*Gamma1*Pc)*nabla_terms["dlngdlnP"]
				deviation -= nabla_terms["nd_grav"]
			else:
				nabla_terms["dlngdlnP"] = 0.
				nabla_terms["nd_grav"] = 0.

		# rotational term
		if self.simd_userot:
			if Tc > self.mintemp and not self.simd_suppress:
				nabla_terms["nd_rot"] = 4.*(2./3.)**0.5/(self.grav*dens)*omega**2/delta
				deviation += nabla_terms["nd_rot"]
			else:
				nabla_terms["nd_rot"] = 0.

		if self.nablarat_crit and (abs(deviation)/hydrograd > self.nablarat_crit):
			raise AssertionError("ERROR: Hit critical nabla!  Code is now designed to throw an error so you can jump to the point of error.")

		totalgrad = hydrograd + deviation	# Assume magnetic gradient is the same at R = 0 as is here
		temp = Tc + Tc/Pc*totalgrad*(P - Pc)		

		# If we hit the temperature floor, artificially force temp to remain at it, and set integration to isothermal
		if temp <= self.mintemp:
			temp = self.mintemp
			isotherm = True
		else:
			isotherm = False

		return [R, P, temp, Bfld, Pchi, hydrograd, totalgrad, nabla_terms, isotherm]


############################################# INTEGRATOR ###############################################


	def integrate_star(self, dens_c, temp_c, omega, recordstar=False, P_end_ratio=1e-8, ps_eostol=1e-8, outputerr=False):
		"""
		ODE solver that generates stellar profile given a central density dens_c, central 
		temperature temp_c, and solid-body angular speed omega.  All arguments not listed 
		below are same as in __init__().

		Parameters
		----------
		dens_c : central density (g/cm^3)
		temp_c : central temperature (K)
		omega : solid-body angular speed (rad/s)
		outputerr : output any error codes received from integrator
		"""

		stepsize = max(self.mass_want,0.4*self.Msun)*0.01*min(self.mass_tol, 1e-6)	# Found out the hard way that if you use simd_usegammavar, having a large mass_tol can cause errors

#		if ps_eostol > 1e-8:
#			print "WARNING: ps_eostol is now {0:.3e}".format(ps_eostol)

		outerr_code = False		# Integrator error code (see loop)

		# Print out the M = 0 step
		M = 0
		R = 0
		temp = temp_c
		dens = dens_c
		[Pc, entropy] = self.getpress_rhoT(dens, temp)		# P IS THE **GAS** PRESSURE, NOT THE TOTAL PRESSURE!
		Pend = Pc*P_end_ratio
		if recordstar:	# Auto-clears any previous data dict
			self.data = {"M": [M],
			"R": [R],
			"Pgas": [Pc],
			"rho": [dens],
			"T": [temp],
			"Sgas": [entropy],
			"Pmag": [],	# Pmag, B will be populated in the next step (done in case B(r = 0) = infty)
			"B": [],
			"nabla_hydro": [],
			"nabla_mhdr": [],
			"nabla_terms": []}

		errtrig = [0]		# Errortrig to stop integrator when Helmholtz NR-method fails to properly invert the EOS (usually due to errors slightly beyond tolerance)

		# Take one step forward (we know the central values, and we assume dP_chi/dr = 0), assuming the central density does not change significantly
		# first_deriv also returns the starting value of isotherm
		M = stepsize
		R, P, temp, Bfld, Pmag, hydrograd, totalgrad, nabla_terms, isotherm = self.first_deriv(dens, M, Pc, temp, omega, failtrig=errtrig)
		[dens, entropy] = self.getdens_PT(P, temp, failtrig=errtrig)

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
			self.data["Pmag"].append(Pmag)
			self.data["B"].append(Bfld)
			self.data["nabla_hydro"].append(hydrograd)
			self.data["nabla_mhdr"].append(totalgrad)
			self.data["nabla_terms"].append(nabla_terms)

		# Continue stepping using scipy.integrate.odeint
		while P > Pend:

			# Adaptive stepsize
			y_in = np.array([R, P, temp])
			# The only way to "self-consistently" print out B is in the derivative, where it could be modified to force B^2/Pgas to be below some critical value
			[dy, Bfld, Pmag, hydrograd, totalgrad, nabla_terms] = self.derivatives(y_in, M, omega, ps_eostol=ps_eostol, m_step=stepsize, isotherm=isotherm, grad_full=True)
			stepsize = self.stepcoeff*min(abs(y_in/(dy+1e-35)))

			if recordstar:
				self.data["Pmag"].append(Pmag)
				self.data["B"].append(Bfld)
				self.data["nabla_hydro"].append(hydrograd)
				self.data["nabla_mhdr"].append(totalgrad)
				self.data["nabla_terms"].append(nabla_terms)

			R, P, temp = scipyinteg.odeint(self.derivatives, y_in, np.array([M,M+stepsize]), 
												args=(omega, errtrig, ps_eostol, stepsize, isotherm), h0=stepsize*0.01, hmax = stepsize, mxstep=1000)[1,:]

			if temp <= self.mintemp and not isotherm:
				R, P, temp, M = self.connect_isotherm(y_in, M, stepsize, omega, Pend, errtrig, ps_eostol, isotherm)
				isotherm = True
			else:
				M += stepsize

			[dens, entropy] = self.getdens_PT(P, temp, failtrig=errtrig)

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
		y_in = np.array([R, P, temp])
		[dy, Bfld, Pmag, hydrograd, totalgrad, nabla_terms] = self.derivatives(y_in, M, omega, ps_eostol=ps_eostol, m_step=stepsize, isotherm=isotherm, grad_full=True)

		# Generate a fake data point where M is exactly mass_want.  Useful for setting initial conditions for 3D WD simulations.
		if recordstar:
			self.data["Pmag"].append(Pmag)
			self.data["B"].append(Bfld)
			self.data["nabla_hydro"].append(hydrograd)
			self.data["nabla_mhdr"].append(totalgrad) 
			self.data["nabla_terms"].append(nabla_terms) 

			if self.fakeouterpoint:
				stepsize = self.stepcoeff*min(abs(y_in/(dy+1e-30)))
				y_out = scipyinteg.odeint(self.derivatives, y_in, np.array([M,M+stepsize]), args=(omega, errtrig, ps_eostol, stepsize, isotherm), h0=stepsize*0.01, hmax = stepsize)[1,:]

				self.data["M"].append(max(self.mass_want, M+stepsize))
				self.data["R"].append(y_out[0])
				self.data["Pgas"].append(0.)
				self.data["rho"].append(0.)
				self.data["T"].append(0.)
				self.data["Sgas"].append(0.)
				y_in = y_out	# Gradients one last time
				[dy, Bfld, Pmag, hydrograd, totalgrad, nabla_terms] = self.derivatives(y_in, M, omega, ps_eostol=ps_eostol, m_step=stepsize, isotherm=isotherm, grad_full=True)
				self.data["Pmag"].append(Pmag)
				self.data["B"].append(Bfld)
				self.data["nabla_hydro"].append(hydrograd)
				self.data["nabla_mhdr"].append(totalgrad)
				self.data["nabla_terms"].append(nabla_terms)

			self.unpack_nabla_terms()
			for item in self.data.keys():
				self.data[item] = np.array(self.data[item])

		if outputerr:
			return [M, outerr_code]
		else:
			return M

	def connect_isotherm(self, y_in, M, stepsize, omega, Pend, errtrig, ps_eostol, isotherm, iter_max=1000, subtol=1e-8):
		"""
		Loop to achieve self.mintemp within tolerance self.mintemp_reltol
		"""
		substep = 0.1*stepsize			# we'll use constant stepsizes
		ys_in = np.array(y_in)			# arrays are passed by reference, not by value!
		P = ys_in[1]
		sM = M
		i = 0
		while i < iter_max and P > Pend and abs(substep) > subtol*stepsize:			# Integrate forward toward self.mintemp
			R, P, temp = scipyinteg.odeint(self.derivatives, ys_in, np.array([sM,sM+substep]), 
												args=(omega, errtrig, ps_eostol, substep, isotherm), h0=substep*0.01, hmax = substep, mxstep=1000)[1,:]
			sM += substep

			#print R, P, temp, sM, substep/stepsize

			# If we're within the temperature tolerance
			if abs(temp - self.mintemp)/self.mintemp < self.mintemp_reltol:
#				if self.verbose:
#					print ">>>Isotherm: current M = {2:.6e}, temp = {0:.6e}, deviation {1:.6e} from self.mintemp. Switching to isothermal.".format(temp, abs(temp - self.mintemp)/self.mintemp, sM)
				break

			# If we've overshot
			if temp < self.mintemp:
				R, P, temp = np.array(ys_in)	# reset R, P, temp to start of step
				sM -= substep					# reverse mass step
				substep = 0.1*substep			# repeat last integration step with greater accuracy

			# If we've done neither, set up next integration
			ys_in = np.array([R, P, temp])
				
		return [R, P, temp, sM]


	def unpack_nabla_terms(self):
		len_nabla = len(self.data["nabla_terms"])
		for item in self.data["nabla_terms"][0].keys():
			self.data[item] = np.zeros(len_nabla)
		for i in range(len_nabla):
			for item in self.data["nabla_terms"][0].keys():
				self.data[item][i] = self.data["nabla_terms"][i][item]
		del self.data["nabla_terms"]	#delete nabla_terms to remove duplicate copies


########################################### POSTPROCESSING FUNCTIONS #############################################


	def weightedavg(self, item, kind="mass"):
		if kind == "mass":
			return scipyinteg.simps(4.*np.pi*self.data["R"]**2*self.data["rho"]*self.data[item], x=self.data["R"])/scipyinteg.simps(4.*np.pi*self.data["R"]**2*self.data["rho"], x=self.data["R"])
		else:
			return scipyinteg.simps(4.*np.pi*self.data["R"]**2*self.data[item], x=self.data["R"])/scipyinteg.simps(4.*np.pi*self.data["R"]**2, x=self.data["R"])


	@staticmethod
	def getIfunction(r, rho):
		"""Returns cumulative moment of inertia I as a function of r (the WD moment of inertial is the largest I returned).
		"""
		interp = interp1d(r,r**4*rho,kind = 'linear')


	@staticmethod
	def getmomentofinertia(r, rho):
		"""Obtains moment of inertia from density-radius relation.
		"""

		# Gives roughly same error, but more computational force, so don't bother:
		#r_aug = concatenate([array([0.]), r, np.array([1e11])])
		#rho_aug = concatenate([array([rho[0]]), rho, np.array([0.])])
		#interp = interp1d(r_aug,r_aug**4*rho_aug,kind = 'slinear')
		#integ_out = 8.*np.pi/3.*scipyinteg.quadrature(interp, 0, max(r))
	
		return 8.*np.pi/3.*scipyinteg.cumtrapz(r**4*rho, x=r, initial=0.)


	@staticmethod
	def cumsimps(y, x=None, even='avg'):
		"""Cumulative Simpsons rule integration (in the vein of numpy.integrate.cumtrapz)
		"""
		outarr = np.zeros(len(x))
		for i in range(1,len(x)):
			outarr[i] = scipyinteg.simps(y[:i+1], x=x[:i+1], even=even)
		return outarr


	def getenergies(self, togglecoulomb=True):
		"""Obtains specific energies (lower case e), total energies (upper case E) and soundspeeds
		"""

		self.data["eint"] = np.zeros(len(self.data["rho"]))
		self.data["edeg"] = np.zeros(len(self.data["rho"]))
		self.data["c_s"] = np.zeros(len(self.data["rho"]))
		for i in range(len(self.data["rho"])):
#			pressure,self.data["eint"][i],self.data["c_s"][i],gammaout,entropy,dummyfail = myhmag.gethelmholtzeos(self.data["T"][i],self.data["rho"][i],self.abar,self.zbar,togglecoulomb)
#			pressure,self.data["edeg"][i],soundspeed,gammaout,entropy,dummyfail = myhmag.gethelmholtzeos(1000.,self.data["rho"][i],self.abar,self.zbar,togglecoulomb)
			[self.data["eint"][i], self.data["edeg"][i], self.data["c_s"][i]] = self.gethelmeos_energies(self.data["rho"][i], self.data["T"][i], togglecoulomb=togglecoulomb)
		self.data["eth"] = self.data["eint"] - self.data["edeg"]
		self.data["erot"] = (2./3.)*self.data["R"]**2*self.omega**2		# specific rotational energy I*omega^2/m = 2/3*r^2*omega^2

		self.data["Edeg"] = 4.*np.pi*scipyinteg.cumtrapz(self.data["edeg"]*self.data["R"]**2*self.data["rho"], x=self.data["R"], initial=0.)
		self.data["Eth"] = 4.*np.pi*scipyinteg.cumtrapz(self.data["eth"]*self.data["R"]**2*self.data["rho"], x=self.data["R"], initial=0.)
		# Binding energy E_pot = \int_0^m -GM_enc/r dm (http://farside.ph.utexas.edu/teaching/301/lectures/node153.html)
		self.data["Epot"] = -4.*np.pi*self.grav*scipyinteg.cumtrapz(self.data["M"]*self.data["R"]*self.data["rho"], x=self.data["R"], initial=0.)
		self.data["Erot"] = 0.5*self.getmomentofinertia(self.data["R"], self.data["rho"])*self.omega**2
		self.data["eb"] = self.data["B"]**2/8./np.pi/self.data["rho"]	# B^2/8pi is the magnetic energy density
		self.data["EB"] = 0.5*scipyinteg.cumtrapz(self.data["B"]**2*self.data["R"]**2, x=self.data["R"], initial=0.)


	def getgradients(self, togglecoulomb=True):
		"""Obtains magnetohydrostatic gradients for diagnosing stellar structures.  Requires a profile to have already been made (but can be run on partially complete sets).
		"""

		len_arr = len(self.data["rho"])
		self.data["dy"] = np.zeros([len_arr,3])
		self.data["B_natural"] = np.zeros(len_arr)
		self.data["Pmag_natural"] = np.zeros(len_arr)
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
#			self.data["B_natural"][i] = self.magf.fBfld(self.data["R"][i], self.data["M"][i])
#			self.data["Pmag_natural"][i] = (1./8./np.pi)*self.magf.fBfld(self.data["R"][i], self.data["M"][i])**2	#Get what user wants B(r) to be
			y_in = np.array([self.data["R"][i], self.data["Pgas"][i], self.data["T"][i]])
			[self.data["dy"][i], Bfld, Pmag, hydrograd, totalgrad, nabla_terms] = self.derivatives(y_in, self.data["M"][i], self.omega, m_step=m_step[i], grad_full=True)
			[adgradred, hydrograd, self.data["nu"][i], self.data["alpha"][i], self.data["delta"][i], self.data["gamma_ad"][i], self.data["cP"][i], cPydro_dumm, c_s_dumm] = self.geteosgradients(self.data["rho"][i], self.data["T"][i], Pmag)
			self.data["nabla_ad"][i] = (self.data["Pgas"][i] + Pmag)/self.data["T"][i]*adgradred
		self.data["dy"][0] = np.array(self.data["dy"][1])		#derivatives using standard function are undefined at R = 0.


	def getconvection(self, togglecoulomb=True, fresh_calc=False):
		"""Obtains convective structure, calculated using a combination of Eqn. 9 of Piro & Chang 08 and modified mixing length theory (http://adama.astro.utoronto.ca/~cczhu/runawaywiki/doku.php?id=magderiv#modified_limiting_cases_of_convection).  Currently doesn't account for magnetic energy in any way, so may not be consistent with MHD stars.
		"""

		# Obtain energies and additional stuff
		if fresh_calc or not self.data.has_key("Epot"):
			self.getenergies()

		if fresh_calc or not self.data.has_key("dy"):
			self.getgradients()

		R = np.array(self.data["R"])
		R[0] = max(1e-30, R[0])
		dPdr = self.data["dy"][:,1]/self.data["dy"][:,0]
		self.data["agrav"] = -dPdr/self.data["rho"]			# used to just be self.grav*self.data["M"]/R**2; now this is "agrav_norot" below
		self.data["agrav_norot"] = self.grav*self.data["M"]/R**2
		self.data["H_P"] = -self.data["Pgas"]/dPdr			# Assuming this is the GAS-ONLY SCALE HEIGHT!
		self.data["H_P"][0] = self.data["H_P"][1]			# Removing singularity at zero
		HPred = (self.data["Pgas"]/self.grav/self.data["rho"]**2)**0.5	# Reduced version that includes damping of H_P near r = 0; used for nabla calculations in derivatives
		self.data["H_Preduced"] = np.array(self.data["H_P"])
		self.data["H_Preduced"][HPred/self.data["H_P"] <= 1] = HPred[HPred/self.data["H_P"] <= 1]

		# obtain epsilon_nuclear from Marten's MESA tables
		self.data["eps_nuc"] = np.zeros(len(self.data["rho"]))
		td = rtc.timescale_data(max_axes=[1e12,1e12])
		eps_nuc_interp = td.getinterp2d("eps_nuc")
		for i in range(len(self.data["eps_nuc"])):
			self.data["eps_nuc"][i] = eps_nuc_interp(self.data["rho"][i], self.data["T"][i])

		# obtain convective luminosity
		self.data["Lnuc"] = 4.*np.pi*scipyinteg.cumtrapz(self.data["eps_nuc"]*R**2*self.data["rho"], x=R, initial=0.)
		self.data["Lconv"] = self.data["Lnuc"]*(1. - self.data["Eth"]/max(self.data["Eth"])*max(self.data["Lnuc"])/self.data["Lnuc"])
		self.data["Lconv"][0] = 0.
		self.data["Fnuc"] = self.data["Lnuc"]/4./np.pi/R**2
		self.data["Fconv"] = self.data["Lconv"]/4./np.pi/R**2
		self.data["vconv"] = (self.data["delta"]*self.data["agrav"]*self.data["H_Preduced"]/self.data["cP"]/self.data["T"]*self.data["Fconv"]/self.data["rho"])**(1./3.)
		self.data["vconv"][0] = max(self.data["vconv"][0], self.data["vconv"][1])
		self.data["vnuc"] = (self.data["delta"]*self.data["agrav"]*self.data["H_Preduced"]/self.data["cP"]/self.data["T"]*self.data["Fnuc"]/self.data["rho"])**(1./3.)	# Equivalent convective velocity of entire nuclear luminosity carried away by convection
		self.data["vnuc"][0] = max(self.data["vnuc"][0], self.data["vnuc"][1])


	def gettimescales(self, fresh_calc=False):
		"""Obtains WWK04 integral and relevant timescales for convective transport, central nuclear burning, etc.
		"""

		if fresh_calc or not self.data.has_key("vnuc"):
			self.getconvection(fresh_calc=fresh_calc)

		# Obtain Woosley, Wunch & Kuhlen 2004 Eqn. 36. integral
		dTdr = self.data["T"]/self.data["Pgas"]*self.data["nabla_mhdr"]*self.data["dy"][:,1]/self.data["dy"][:,0]
		self.data["WWK04_integral"] = scipyinteg.cumtrapz(dTdr + self.data["eps_nuc"]/self.data["cP"]/self.data["vconv"], x=self.data["R"], initial=0.)

		self.data["tau_cc"] = self.data["cP"]*self.data["T"]/self.data["eps_nuc"]								# Nuclear burning timescale
		self.data["t_blob"] = scipyinteg.cumtrapz(1./self.data["vconv"], x=self.data["R"], initial=0.)			# Blob travel time from r = 0 to r = R_WD
		self.data["t_heat_cum"] = scipyinteg.cumtrapz(4.*np.pi*self.data["R"]**2*self.data["rho"]*self.data["cP"]*self.data["T"], x=self.data["R"], initial=0.)/self.data["Lnuc"]											# Cumulative convective heating time (from CWvK unpublished Eqn. 8)
		self.data["t_heat_cum"][0] = 0.									# remove the NaN
		self.data["t_dyn"] = scipyinteg.cumtrapz(1./self.data["c_s"], x=self.data["R"], initial=0.)	# Sound crossing (dynamical) time
		self.data["t_heat"] = max(self.data["t_heat_cum"])				# Convective heating time for entire star
		i = min((self.data["Lnuc"]/max(self.data["Lnuc"]) > 0.95).nonzero()[0])
		self.data["R_nuc"] = self.data["R"][i]							# Outer boundary of nuclear burning region
		self.data["t_conv"] = self.data["t_blob"][i]					# Blob travel time across nuclear burning region (CWvK unpublished Eqn. 25)


########################################### BLANK STARMOD FOR POST-PROCESSING #############################################

	@classmethod
	def blankstar(cls, input_data, i_args=False, backcalculate=False, **kwargs):
		"""Generates a dummy star, and loads in user-inputted stellar data.  Data structure MUST include "R", "M", "rho", "T" and "B" for self.backcalculate to work.  Arguments are otherwise identical to those of maghydrostar.__init__; for kwargs, see class constructor arguments.

		Arguments:
		input_data: struct of data that MUST include M and T, but otherwise
		i_args: input arguments to maghydrostar from runaway.py/make_runaway
		backcalculate: backcalculate gradients and convective values
		**kwargs: any additional arguments passable to class initialization function can be passed here.
		"""

		if i_args:
			dataobj = cls(max(input_data["M"]), input_data["T"][0], magprofile=i_args["magprofile"], omega=i_args["omega"], S_want=False, mintemp=i_args["mintemp"], composition=i_args["composition"], derivtype=i_args["derivtype"], simd_userot=i_args["simd_userot"], simd_usegammavar=i_args["simd_usegammavar"], simd_usegrav=i_args["simd_usegrav"], simd_suppress=i_args["simd_suppress"], nablarat_crit=False, P_end_ratio=i_args["P_end_ratio"], ps_eostol=i_args["ps_eostol"], fakeouterpoint=i_args["fakeouterpoint"], stop_invertererr=i_args["stop_invertererr"], stop_mrat=i_args["stop_mrat"], stop_positivepgrad=i_args["stop_positivepgrad"], densest=False, mass_tol=i_args["mass_tol"], omega_crit_tol=i_args["omega_crit_tol"], nreps=100, stopcount_max=5, dontintegrate=True, verbose=False)
		else:
			dataobj = cls(max(input_data["M"]), input_data["T"][0], dontintegrate=True, **kwargs)

		dataobj.data = {}
		for item in input_data.keys():
			dataobj.data[item] = copy.deepcopy(input_data[item])	#though shallow copying would have worked too

		if backcalculate:
			dataobj.backcalculate()

		return dataobj


	def backcalculate(self, togglecoulomb=True, fresh_calc=False):
		"""For situations where user-inputted R, M, rho, T and B are given, back-calculate.
		"""

		# Check if required values are there.
		if not np.prod(np.array([self.data.has_key(x) for x in ["R", "M", "rho", "T", "B"]])):
			print "ERROR - one of R, M, rho, T or B not found in user input!  Exiting."
			return

		# Legacy values check.
		if self.data.has_key("P"):
			self.data["Pgas"] = self.data["P"]

		if self.data.has_key("S"):
			self.data["Sgas"] = self.data["S"]

		datalength = len(self.data["M"])

		# But if we don't seem to have the standard faire values always printed out, just re-calculate them (yes, that means we do overwrite the ones we do have)
		if not np.prod(np.array([self.data.has_key(x) for x in ["Pgas", "Sgas", "Pmag", "nabla_hydro", "nabla_mhdr"]])):
			print "WARNING - I didn't find Pgas, Sgas, Pmag, nabla_hydro, or nabla_mhdr in your input - recalculating (including deviations)!"
			
			m_step = self.data["M"][1:] - self.data["M"][:-1]
			m_step = np.concatenate( [m_step, np.array([m_step[-1]])] )

			# Create temporary housing for data
			temp_data = {"Pgas": np.zeros(datalength),
						"Sgas": np.zeros(datalength),
						"Pmag": np.zeros(datalength),
						"nabla_hydro": np.zeros(datalength),
						"nabla_mhdr": np.zeros(datalength),
						}
			if self.simd_usegammavar:
				self.data["dlngamdlnP"] = np.zeros(datalength)
				self.data["nd_gamma"] = np.zeros(datalength)
			if self.simd_usegrav:
				self.data["dlngdlnP"] = np.zeros(datalength)
				self.data["nd_grav"] = np.zeros(datalength)
			if self.simd_userot:
				self.data["nd_rot"] = np.zeros(datalength)

			for i in range(datalength):
				[temp_data["Pgas"][i], temp_data["Sgas"][i]] = self.getpress_rhoT(self.data["rho"][i], self.data["T"][i])
				[dydx, Bfld, temp_data["Pmag"], temp_data["nabla_hydro"], temp_data["nabla_mhdr"], nabla_terms]
				y_in = np.array([self.data["R"][i], self.data["Pgas"][i], self.data["T"][i]])
				[dummydy, dummyBfld, temp_data["Pmag"], temp_data["nabla_hydro"], temp_data["nabla_mhdr"], nabla_terms] = self.derivatives(y_in, self.data["M"][i], self.omega, m_step=m_step[i], grad_full=True)

				if self.simd_usegammavar:
					self.data["dlngamdlnP"][i] = nabla_terms["dlngamdlnP"]
					self.data["nd_gamma"][i] = nabla_terms["nd_gamma"]
				if self.simd_usegrav:
					self.data["dlngdlnP"][i] = nabla_terms["dlngdlnP"]
					self.data["nd_grav"][i] = nabla_terms["nd_grav"]
				if self.simd_userot:
					self.data["nd_rot"][i] = nabla_terms["nd_rot"]

			for item in temp_data.keys():
				if not self.data.has_key(item):
					self.data[item] = copy.deepcopy(temp_data[item])

		self.gettimescales(fresh_calc=fresh_calc)
