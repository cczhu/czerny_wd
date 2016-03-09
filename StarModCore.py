import numpy as np
import pylab
from scipy.interpolate import interp1d
import scipy.integrate as scipyinteg
import os, sys
import copy as cp
import magprofile_mass as magprof
import myhelm_magstar as myhmag
import rhoTcontours as rtc


class maghydrostar_core():
	"""
	Hydrostatic star generator core functionality, containing shooters, equation
	of state functions and common post-processing functions.  Parent class of 
	maghydrostar and mhs_steve; see child class documentation for more details.

	def __init__(self, mass, temp_c, magprofile=False, omega=0., Lwant=0., 
				mintemp=1e5, composition="CO", togglecoulomb=True,
				fakeouterpoint=False, stop_invertererr=True, 
				stop_mrat=2., stop_positivepgrad=True, stop_mindenserr=1e-10, 
				mass_tol=1e-6, L_tol=1e-6, omega_crit_tol=1e-3, nreps=30, 
				stopcount_max=5, verbose=True):

	Parameters
	----------
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
	Lwant : wanted angular momentum.
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

	Returns
	-------
	mscore : maghydrostar_core class instance
	"""

	def __init__(self, mass, temp_c, magprofile=False, omega=0., Lwant=0., 
				mintemp=1e5, composition="CO", togglecoulomb=True,
				fakeouterpoint=False, stop_invertererr=True, 
				stop_mrat=2., stop_positivepgrad=True, stop_mindenserr=1e-10, 
				mass_tol=1e-6, L_tol=1e-6, omega_crit_tol=1e-3, nreps=30, 
				stopcount_max=5, verbose=True):

		# Constants
		self.Msun = 1.9891e33
		self.grav = 6.67384e-8

		# Stellar properties
		self.mass_want = mass
		self.temp_c = temp_c
		self.L_want = Lwant

		# Integration settings
		self.stepcoeff = 1e-2
		self.nreps = nreps
		self.fakeouterpoint = fakeouterpoint
		self.stop_invertererr = stop_invertererr
		self.stop_mrat = stop_mrat
		self.stop_positivepgrad = stop_positivepgrad
		self.stop_mindenserr = stop_mindenserr
		self.s_mind_errcode = False				# If True, returns mindenserr as an outerr_code (usually unwanted behaviour)
		self.stopcount_max = stopcount_max

		# Integration tolerances
		self.mass_tol = mass_tol
		self.omega_crit_tol = omega_crit_tol
		self.L_tol = L_tol

		# Remember to print messages
		self.verbose = verbose

		# EOS settings
		self.togglecoulomb = togglecoulomb
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

		# Run a dummy gethelmeos to check if EOS is initialized:
		pressure,energy,soundspeed,gammaout,entropy,checkeosfailure = myhmag.gethelmholtzeos(1e5,1e2,2.,4.,True)
		if checkeosfailure:					# If 1 is returned...
			print "I noticed you haven't initialized helmholtz.  Doing so now."
			myhmag.initializehelmholtz()	# ...initialize helmholtz

		# Magnetic profile
		if not magprofile:	# If magprofile is false, generate a zero-field profile
			if self.verbose:
				print "magprofile == False - will assume star has no magnetic field!"
			self.magf = magprof.magprofile(None, None, None, None, blankfunc=True)
			self.nabladev = False
		elif (type(magprofile) == float or type(magprofile) == np.float64):		# If magprofile is constant nabladev = delta = B^2/(B^2 + 4pi*Gamma1*Pgas)
			self.nabladev = magprofile
			if self.nabladev <= 0.:
				raise AssertionError("ERROR: if you use a constant delta field, then magprofile must be a positive value!")
			if self.verbose:
				print "Derivative WITH CONSTANT DEVIATION RATIO {0:.3e} selected! Using self.nabladev as deviation delta(r) (MM09 Eqn. 4).".format(self.nabladev)
			self.magf = False
		else:
			self.magf = magprofile
			self.nabladev = False

		# Temperature floor
		self.mintemp = mintemp
		self.mintemp_reltol = 1e-6	# Relative tolerance to shoot for in connect_isotherm
		if self.verbose:
			print "Minimum temperature set to {0:.3e}".format(self.mintemp)

		# Omega input options
		if omega < 0.:
			if self.verbose:
				print "Omega < 0 - max rotation estimator selected!"
			self.omega = omega
		else:
			if self.verbose:
				print "Omega = {0:.3e}".format(omega)
			self.omega = max(0., omega)

		# Initialze data storage dict
		self.data = {}


######################################### SHOOTING ALGORITHMS #########################################

############################## GENERIC STUFF ###############################

	def getcritrot(self, M, R):
		"""Returns critical Omega given M and R.
		"""
		return np.sqrt(M*self.grav/R**3)

############################## 1D MASS SHOOTER ###############################

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


############## NESTED 1D JACOBIAN MASS/ANG MO SHOOTER (SLOWER) ##################


#	def getrotatingstarmodel(self, densest=False, omegaest=False, S_want=False, P_end_ratio=1e-8, ps_eostol=1e-8, damp_nrstep=1, deltastepcoeff=0.05, 
#									interior_dscoeff=0.1, omega_warn=10., out_search=False):
#		"""
#		Newton-Raphson solver to obtain rigidly-rotating WD with user-specified mass and 
#		ANGULAR MOMENTUM.  If you wish to create a WD with user-specified mass and omega, 
#		use self.getstarmodel().  Arguments not defined below have identical meanings as 
#		class initialization ones.

#		Parameters
#		----------
#		damp_nrstep : damping term for Newton-Raphson stepping.  Defaults to 1, 
#			but that will likely lead to overshooting near critical Omega values.
#		deltastepcoeff: when estimating the Jacobian, Delta Omega = 
#			deltastepcoeff*abs(deltaOmega_previous).  Defaults to 0.05.
#		interior_dscoeff: deltastepcoeff for getstarmodel mass-finding sub-loops.  
#			Defaults to 0.1.
#		omega_warn: stop integration if self.omega approaches omega_warn*omega_crit 
#			estimate.  Defaults to 10 to prevent premature stoppage.
#		out_search: prints L, omega calculated by integrate_star during the 
#			shooting process.
#		"""

#		if self.L_want <= 0.:
#			print "Please define self.L_want before using this function!  Quitting."
#			return

#		# If user doesn't specify, randomly pick a relatively safe omega
#		if omegaest:
#			self.omega = omegaest
#		else:
#			self.omega = 0.2

#		if self.verbose:
#			print "==== GENERATING NEW ROTATING STAR MODEL ===="

#		i = 0
#		outerr_code = self.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, deltastepcoeff=interior_dscoeff)	# First shot
#		Ltot = self.getmomentofinertia(self.data["R"], self.data["rho"])[-1]*self.omega

#		if self.verbose:
#			print "Omega overloop first shot: L = {0:.6e} (vs. L_want = {1:.6e}; relative err. {2:.6e}) ".format( \
#															Ltot, self.L_want, abs(Ltot - self.L_want)/self.L_want)

#		delta_omega = self.omega		# Note to self - this shallow-copies self.omega, which is a float (and not some weird class instance thing)

#		# Keep past shooting attempts
#		L_arr = np.array([Ltot])
#		omega_arr = np.array([self.omega])

#		while abs(Ltot - self.L_want) >= self.L_tol*self.L_want and i < self.nreps and not outerr_code:

#			if self.verbose:
#				print "== RUNNING GETSTARMODEL WITHIN GETJACOBIAN_OMEGA =="

#			[dLdomega, outerr_code] = self.getjacobian_omega(self.data["rho"][0], Ltot, deltastepcoeff*delta_omega, S_want=S_want,
#													P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, interior_dscoeff=interior_dscoeff)		# Calculate Jacobian (self.omega auto-passed)
#			delta_omega = damp_nrstep*(self.L_want - Ltot)/dLdomega
#			self.omega += delta_omega

#			if self.omega < 0.:		#  Negative omega is not defined
#				self.omega = abs(delta_omega)*0.1
#			if self.verbose:
#				print "dLdomega is {0}, delta_omega is {1}".format(dLdomega, delta_omega)
#				print "Old omega = {0:.6e}; new omega = {1:.6e}".format(self.omega - delta_omega, self.omega)
#				print "== RUNNING GETSTARMODEL WITH REVISED OMEGA =="

#			self.getstarmodel(densest=self.data["rho"][0], S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, deltastepcoeff=interior_dscoeff)
#			Ltot = self.getmomentofinertia(self.data["R"], self.data["rho"])[-1]*self.omega
#			if self.verbose:
#				print "Omega overloop shot: L = {0:.6e} (vs. L_want = {1:.6e}; relative err. {2:.6e}) ".format(Ltot, 
#																self.L_want, abs(Ltot - self.L_want)/self.L_want)
#			L_arr = np.append(L_arr, Ltot); omega_arr = np.append(omega_arr, self.omega)

#			if self.omega > omega_warn*self.getcritrot(max(self.data["M"]), self.data["R"][-1]):
#				outerr_code = "Omega/Omega_crit > {0:.1e}".format(omega_warn)

#			i += 1

#		if i == self.nreps:
#			print "WARNING, maximum number of shooting attempts {0:d} reached!".format(i)

#		# If we incur any error
#		if outerr_code:
#			print "OUTERR_CODE {0:s}!  EXITING FUNCTION!".format(outerr_code)
#			return outerr_code

#		if (abs(Ltot - self.L_want)/self.L_want > self.L_tol):
#			print "ERROR! (L_tot - L_want)/L_want = {0:.6e}   THIS IS BIGGER THAN YOUR TOLERANCE!  CHECK YOUR ICS!".format( \
#																(Ltot - self.L_want)/self.L_want)
#		else:
#			print "(L_tot - L_want)/L_want = {0:.6e}".format((Ltot - self.L_want)/self.L_want)

#		if out_search:
#			return [L_arr, omega_arr]


#	def getjacobian_omega(self, densest, Ltot, domega, S_want=False, P_end_ratio=1e-8, ps_eostol=1e-8, interior_dscoeff=0.1):
#		"""Returns dL/domega for getrotatingstarmodel()
#		"""
#		omega_alt = self.omega + domega
#		outerr_code = self.getstarmodel(densest=densest, omega_user=omega_alt, S_want=S_want, 
#										P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, deltastepcoeff=interior_dscoeff)
#		Ltot_domega = self.getmomentofinertia(self.data["R"], self.data["rho"])[-1]*omega_alt
#		return [(Ltot_domega - Ltot)/domega, outerr_code]


############### 2D JACOBIAN MASS/ANG MO SHOOTER (FASTER, MORE DANGEROUS) ####################


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


################ OVERLOOP OF GETSTARMODEL FOR OBTAINING CRITICAL OMEGA ####################


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


################# OVERLOOP OF GETSTARMODEL FOR OBTAINING MAGNETIC FIELD CONFIGS ##############


	def getBcnabladevmodel(self, Bc_want, nabladev_est=0.001, Bc_tol=1e-4, mass_user=False, densest=False, S_want=False, P_end_ratio=1e-8, ps_eostol=1e-8, deltastepcoeff=0.1, out_search=False):
		"""
		Wrapper to obtain WD with constant delta = B^2/(B^2 + 4pi*Gamma1*Pgas) 
		with user-specified central magnetic field strength Bc.  Calls 
		getstarmodel as a subloop to obtain stars with a specific mass.  
		Arguments not listed below have identical meanings as class 
		initialization ones.

		Parameters
		----------
		Bc_want : user-specified central magnetic field strength
		nabladev_est : initial estimate for nabladev
		Bc_tol : tolerance between calculated and user specified central field
		mass_user : use mass_user instead of self.mass_want in getstarmodel.
		deltastepcoeff : when estimating the Jacobian, Delta rho = 
			deltastepcoeff*abs(deltadens_previous).  Defaults to 0.1.
		out_search : prints the rho and corresponding M values calculated by 
			integrate_star during the shooting process
		"""

		if mass_user:
			self.mass_want = mass_user

		# self.nabladev = 0 means use magf for field profile!
		if nabladev_est <= 0.:
			print "nabladev_est cannot be less than or equal to zero!"
			nabladev_est = 0.001
		self.nabladev = nabladev_est
		deltadev = self.nabladev

		if self.verbose:
			print "== RUNNING getBcnabladevmodel, INITIAL nabladev = {0:.6e}".format(self.nabladev)

		i = 0
		outerr_code = self.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, deltastepcoeff=deltastepcoeff)
		Bc = self.data["B"][0]
		densest = self.data["rho"][0]

		# Keep past shooting attempts
		B_arr = np.array([Bc])
		dev_arr = np.array([self.nabladev])

		while abs(Bc - Bc_want) >= Bc_tol*Bc_want and i < self.nreps and not outerr_code:

			[dBddev, outerr_code] = self.getjacobian_Bdev(densest, Bc, deltastepcoeff*abs(deltadev), 
													S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, deltastepcoeff=deltastepcoeff)	# Calculate Jacobian
			deltadev = (Bc_want - Bc)/dBddev
			self.nabladev += deltadev

			if self.nabladev <= 0.:	# nabladev can't be zero or negative
				self.nabladev = abs(deltadev)*0.1
			if self.verbose:
				print "== dBddev is {0:.6e}, deltadev is {1:.6e}".format(dBddev, deltadev)
				print "== Old dev = {0:.6e}; new dev = {1:.6e}".format(self.nabladev - deltadev, self.nabladev)

			outerr_code = self.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, deltastepcoeff=deltastepcoeff)
			Bc = self.data["B"][0]
			if self.verbose:
				print "== Current shot: Bc = {0:.6e} (vs. Bc_want = {1:.6e}; relative err. {2:.6e})".format(Bc, Bc_want, abs(Bc - Bc_want)/Bc_want)
			B_arr = np.append(B_arr, Bc)
			dev_arr = np.append(dev_arr, self.nabladev)

			i += 1

		if i == self.nreps:
			print "== ERROR, maximum number of shooting attempts {0:d} reached!".format(i)
			return "nrepsnotconverged_err"

		# If we incur any error
		if outerr_code:
			print "== OUTERR_CODE {0:s}!  EXITING FUNCTION!".format(outerr_code)
			return outerr_code

		if abs((Bc - Bc_want)/Bc_want) > Bc_tol:
			print "== ERROR! (Bc - Bc_want)/Bc_want = {0:.6e}  THIS IS BIGGER THAN YOUR TOLERANCE!  CHECK YOUR ICS!".format((Bc - Bc_want)/Bc_want)
		else:
			print "== (Bc - Bc_want)/Bc_want = {0:.6e}".format((Bc - Bc_want)/Bc_want)

		if out_search:
			return [B_arr, dev_arr]


	def getjacobian_Bdev(self, densest, Bc, ddev, S_want=False, P_end_ratio=1e-8, ps_eostol=1e-8, deltastepcoeff=0.1):
		"""Returns dB/dnabladev.
		"""
		self.nabladev += ddev
		outerr_code = self.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, deltastepcoeff=deltastepcoeff)
		Bcn = self.data["B"][0]
		self.nabladev -= ddev
		return [(Bcn - Bc)/ddev, outerr_code]


	def getEBnabladevmodel(self, EBrat_want, nabladev_est=0.001, EBr_tol=1e-4, mass_user=False, densest=False, S_want=False, P_end_ratio=1e-8, ps_eostol=1e-8, deltastepcoeff=0.1, out_search=False):
		"""
		Wrapper to obtain WD with constant delta = B^2/(B^2 + 4pi*Gamma1*Pgas) 
		with user-specified ratio between total magnetic and total energy 
		E_B/E_tot.  Calls getstarmodel as a subloop to obtain stars with a 
		specific mass.  Arguments not listed below have identical meanings as 
		class initialization ones.

		Parameters
		----------
		EBrat_want : user-specified central magnetic field strength
		nabladev_est : initial estimate for nabladev
		EBr_tol : tolerance between calculated and user specified central field
		mass_user : use mass_user instead of self.mass_want in getstarmodel.
		deltastepcoeff : when estimating the Jacobian, Delta rho = 
			deltastepcoeff*abs(deltadens_previous).  Defaults to 0.1.
		out_search : prints the rho and corresponding M values calculated by 
			integrate_star during the shooting process
		"""

		if mass_user:
			self.mass_want = mass_user

		# self.nabladev = 0 means use magf for field profile!
		if nabladev_est <= 0.:
			print "nabladev_est cannot be less than or equal to zero!"
			nabladev_est = 0.001
		self.nabladev = nabladev_est
		deltadev = self.nabladev

		if self.verbose:
			print "== RUNNING getEBnabladevmodel, INITIAL nabladev = {0:.6e}".format(self.nabladev)

		i = 0
		outerr_code = self.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, deltastepcoeff=deltastepcoeff)
		self.getenergies()
		Etot = self.data["Eth"] + self.data["Erot"] + self.data["Edeg"] + self.data["EB"] + self.data["Epot"]
		EBrat = self.data["EB"][-1]/abs(Etot[-1])
		densest = self.data["rho"][0]

		# Keep past shooting attempts
		EBr_arr = np.array([EBrat])
		dev_arr = np.array([self.nabladev])

		while abs(EBrat - EBrat_want) >= EBr_tol*EBrat_want and i < self.nreps and not outerr_code:

			[dEBrddev, outerr_code] = self.getjacobian_EBrdev(densest, EBrat, deltastepcoeff*abs(deltadev), 
													S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, deltastepcoeff=deltastepcoeff)	# Calculate Jacobian
			deltadev = (EBrat_want - EBrat)/dEBrddev
			self.nabladev += deltadev

			if self.nabladev <= 0.:	# nabladev can't be negative
				self.nabladev = abs(deltadev)*0.1
			if self.verbose:
				print "== dBddev is {0:.6e}, deltadev is {1:.6e}".format(dEBrddev, deltadev)
				print "== Old dev = {0:.6e}; new dev = {1:.6e}".format(self.nabladev - deltadev, self.nabladev)

			outerr_code = self.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, deltastepcoeff=deltastepcoeff)
			self.getenergies()
			Etot = self.data["Eth"] + self.data["Erot"] + self.data["Edeg"] + self.data["EB"] + self.data["Epot"]
			EBrat = self.data["EB"][-1]/abs(Etot[-1])
			if self.verbose:
				print "== Current shot: EBrat = {0:.6e} (vs. EBrat_want = {1:.6e}; relative err. {2:.6e})".format(EBrat, EBrat_want, abs(EBrat - EBrat_want)/EBrat_want)
			EBr_arr = np.append(EBr_arr, EBrat)
			dev_arr = np.append(dev_arr, self.nabladev)

			i += 1

		if i == self.nreps:
			print "== ERROR, maximum number of shooting attempts {0:d} reached!".format(i)
			return "nrepsnotconverged_err"

		# If we incur any error
		if outerr_code:
			print "== OUTERR_CODE {0:s}!  EXITING FUNCTION!".format(outerr_code)
			return outerr_code

		if abs(EBrat - EBrat_want)/EBrat_want > EBr_tol:
			print "== ERROR! (EB - EB_want)/EB_want = {0:.6e}  THIS IS BIGGER THAN YOUR TOLERANCE!  CHECK YOUR ICS!".format(abs(EBrat - EBrat_want)/EBrat_want)
		else:
			print "== (EB - EB_want)/EB_want = {0:.6e}".format(abs(EBrat - EBrat_want)/EBrat_want)

		if out_search:
			return [EBr_arr, dev_arr]


	def getjacobian_EBrdev(self, densest, EBrat, ddev, S_want=False, P_end_ratio=1e-8, ps_eostol=1e-8, deltastepcoeff=0.1):
		"""Returns dEBrat/dnabladev.
		"""
		self.nabladev += ddev
		outerr_code = self.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, deltastepcoeff=deltastepcoeff)
		self.getenergies()
		Etotn = self.data["Eth"] + self.data["Erot"] + self.data["Edeg"] + self.data["EB"] + self.data["Epot"]
		EBratn = self.data["EB"][-1]/abs(Etotn[-1])
		self.nabladev -= ddev
		return [(EBratn - EBrat)/ddev, outerr_code]


###################################### EOS STUFF #######################################


	# EOS functions are passed failtrig as a list (since lists are mutable).
	def getpress_rhoT(self, dens, temp, failtrig=[-100]):
		pressure, energy, soundspeed, gammaout, entropy, failtrig[0] = \
				myhmag.gethelmholtzeos(temp, dens, self.abar, self.zbar, self.togglecoulomb)
		return [pressure, entropy]


	def geteosgradients(self, dens, temp, Pchi, failtrig=[-100]):
		adgradred, hydrograd, nu, alpha, delta, Gamma1, cP, cPhydro, c_s, failtrig[0] = \
				myhmag.gethelmgrads(temp, dens, Pchi, self.abar, self.zbar, self.togglecoulomb)
		return [adgradred, hydrograd, nu, alpha, delta, Gamma1, cP, cPhydro, c_s]


	def getpress_rhoS(self, dens, entropy, failtrig=[-100]):
		temp, pressure, energy, soundspeed, gammaout, failtrig[0] = \
				myhmag.geteosinversionsd(dens, self.abar, self.zbar, entropy, self.togglecoulomb)
		return [pressure, temp]

	def getdens_PT(self, press, temp, failtrig=[-100]):
		dens, energy, soundspeed, gammaout, entropy, failtrig[0] = \
				myhmag.geteosinversionpt(temp, self.abar, self.zbar, press, self.togglecoulomb)
		return [dens, entropy]


	def getdens_PS(self, press, entropy, failtrig=[-100]):
		temp, dens, energy, soundspeed, gammaout, failtrig[0] = \
				myhmag.geteosinversionsp(self.abar, self.zbar, press, entropy, True, self.togglecoulomb)
		return [dens, temp]


	def getdens_PS_est(self, press, entropy, failtrig=[-100], dens_est=1e6, temp_est=1e7, eostol=1e-8):
		"""Identical to getdens_PS, but uses user-defined estimates to help with the Newton-Raphson algorithm.
		"""
		temp, dens, energy, soundspeed, gammaout, failtrig[0] = \
				myhmag.geteosinversionsp_withest(self.abar, self.zbar, dens_est, temp_est, press, 
													entropy, True, self.togglecoulomb, eostol)
		return [dens, temp, gammaout]


	def getgamma_PD(self, dens, press, failtrig=[-100]):
		"""Obtains Gamma_1 = dP/drho_ad estimate for first_derivatives functions.
		"""
		temp_dummy, energy_dummy, soundspeed_dummy, Gamma1_est, entropy_dummy, failtrig[0] = \
				myhmag.geteosinversionpd(dens, self.abar, self.zbar, press, self.togglecoulomb)
		return Gamma1_est


	def gethelmeos_energies(self, dens, temp, failtrig=[-100]):
		"""Obtains Gamma_1 = dP/drho_ad estimate for first_derivatives functions.
		"""
		press_dumm, e_int, c_s, gamma_dumm, ent_dumm, dummyfail1 = \
				myhmag.gethelmholtzeos(temp, dens, self.abar, self.zbar, self.togglecoulomb)
		press_dumm, e_deg, sound_dumm, gamma_dumm, ent_dumm, dummyfail2 = \
				myhmag.gethelmholtzeos(1000., dens, self.abar, self.zbar, self.togglecoulomb)
		if dummyfail1 or dummyfail2:
			failtrig[0] = 1
		return [e_int, e_deg, c_s]


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


	def getenergies(self):
		"""Obtains specific energies (lower case e), total energies (upper case E) and soundspeeds
		"""

		self.data["eint"] = np.zeros(len(self.data["rho"]))
		self.data["edeg"] = np.zeros(len(self.data["rho"]))
		self.data["c_s"] = np.zeros(len(self.data["rho"]))
		for i in range(len(self.data["rho"])):
			[self.data["eint"][i], self.data["edeg"][i], self.data["c_s"][i]] = self.gethelmeos_energies(self.data["rho"][i], self.data["T"][i])
		self.data["eth"] = self.data["eint"] - self.data["edeg"]
		self.data["erot"] = (2./3.)*self.data["R"]**2*self.omega**2		# specific rotational energy I*omega^2/m = 2/3*r^2*omega^2

		self.data["Edeg"] = 4.*np.pi*scipyinteg.cumtrapz(self.data["edeg"]*self.data["R"]**2*self.data["rho"], x=self.data["R"], initial=0.)
		self.data["Eth"] = 4.*np.pi*scipyinteg.cumtrapz(self.data["eth"]*self.data["R"]**2*self.data["rho"], x=self.data["R"], initial=0.)
		# Binding energy E_pot = \int_0^m -GM_enc/r dm (http://farside.ph.utexas.edu/teaching/301/lectures/node153.html)
		self.data["Epot"] = -4.*np.pi*self.grav*scipyinteg.cumtrapz(self.data["M"]*self.data["R"]*self.data["rho"], x=self.data["R"], initial=0.)
		self.data["Erot"] = 0.5*self.getmomentofinertia(self.data["R"], self.data["rho"])*self.omega**2
		self.data["eb"] = self.data["B"]**2/8./np.pi/self.data["rho"]	# B^2/8pi is the magnetic energy density
		self.data["EB"] = 0.5*scipyinteg.cumtrapz(self.data["B"]**2*self.data["R"]**2, x=self.data["R"], initial=0.)


	def getconvection(self, td=False, fresh_calc=False):
		"""Obtains convective structure, calculated using a combination of Eqn. 9 of Piro & Chang 08 and modified mixing length theory (http://adama.astro.utoronto.ca/~cczhu/runawaywiki/doku.php?id=magderiv#modified_limiting_cases_of_convection).  Currently doesn't account for magnetic energy in any way, so may not be consistent with MHD stars.
		"""

		# Obtain energies and additional stuff
		if fresh_calc or not self.data.has_key("Epot"):
			self.getenergies()

		if fresh_calc or not self.data.has_key("gamma_ad"):
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
		if not td:
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

		Parameters
		-----------
		input_data: struct of data that MUST include M and T/S.  These can either be 
			arrays or floats.
		i_args: input arguments to maghydrostar from runaway.py/make_runaway
		backcalculate: backcalculate gradients and convective values
		**kwargs: any additional arguments passable to class initialization function 
			can be passed here.

		Examples
		--------
		To load a star with a mass of 1Msun and a dummy central temperature 
		(entropy in the case of mhs_steve):
		>>> mystar = class.blankstar({"M": 1.9891e33, "T": 0.})
		To generate a stellar profile using StarMod.maghydrostar, then read it
		into blankstar:
		>>> mystarold = StarMod.maghydrostar(1.9891e33, 1e8)
		>>> mystar = class.blankstar(myoldstar.data)

		"""
		try:
			blnk_M = max(input_data["M"])
		except:
			blnk_M = input_data["M"]
		if cls.__name__ == "mhs_steve":
			try:
				blnk_therm_c = input_data["Sgas"][0]
			except:
				blnk_therm_c = input_data["Sgas"]
		else:
			try:
				blnk_therm_c = input_data["T"][0]
			except:
				blnk_therm_c = input_data["T"]

		if i_args:
			if cls.__name__ == "mhs_steve":
				dataobj = cls(blnk_M, blnk_therm_c, magprofile=i_args["magprofile"], omega=i_args["omega"], S_want=False, mintemp=i_args["mintemp"], composition=i_args["composition"], togglecoulomb=i_args["tog_coul"], nablarat_crit=False, P_end_ratio=i_args["P_end_ratio"], ps_eostol=i_args["ps_eostol"], fakeouterpoint=i_args["fakeouterpoint"], stop_invertererr=i_args["stop_invertererr"], stop_mrat=i_args["stop_mrat"], stop_positivepgrad=i_args["stop_positivepgrad"], densest=False, mass_tol=i_args["mass_tol"], omega_crit_tol=i_args["omega_crit_tol"], nreps=100, stopcount_max=5, dontintegrate=True, verbose=False)
			else:
				dataobj = cls(blnk_M, blnk_therm_c, magprofile=i_args["magprofile"], omega=i_args["omega"], S_want=False, mintemp=i_args["mintemp"], composition=i_args["composition"], togglecoulomb=i_args["tog_coul"], simd_userot=i_args["simd_userot"], simd_usegammavar=i_args["simd_usegammavar"], simd_usegrav=i_args["simd_usegrav"], simd_suppress=i_args["simd_suppress"], nablarat_crit=False, P_end_ratio=i_args["P_end_ratio"], ps_eostol=i_args["ps_eostol"], fakeouterpoint=i_args["fakeouterpoint"], stop_invertererr=i_args["stop_invertererr"], stop_mrat=i_args["stop_mrat"], stop_positivepgrad=i_args["stop_positivepgrad"], densest=False, mass_tol=i_args["mass_tol"], omega_crit_tol=i_args["omega_crit_tol"], nreps=100, stopcount_max=5, dontintegrate=True, verbose=False)
		else:
			dataobj = cls(blnk_M, blnk_therm_c, dontintegrate=True, **kwargs)

		dataobj.data = {}
		for item in input_data.keys():
			dataobj.data[item] = cp.deepcopy(input_data[item])	#though shallow copying would have worked too

		if backcalculate:
			dataobj.backcalculate()

		return dataobj


	def backcalculate(self):
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

		# But if we don't seem to have the standard faire values always printed out, just re-calculate them (yes, that means we do overwrite the ones we do have)
		if not np.prod(np.array([self.data.has_key(x) for x in ["Pgas", "Sgas", "Pmag", "nabla_hydro", "nabla_mhdr"]])):
			print "WARNING - I didn't find Pgas, Sgas, Pmag, nabla_hydro, or nabla_mhdr in your input - recalculating (including deviations)!"
			
			temp_data = self.backcalculate_subloop()

			for item in temp_data.keys():
				if not self.data.has_key(item):
					self.data[item] = cp.deepcopy(temp_data[item])

		self.gettimescales(fresh_calc=True)
