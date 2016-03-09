import numpy as np
import pylab
from scipy.interpolate import interp1d
import scipy.integrate as scipyinteg
import os, sys
import copy as cp
import magprofile_mass as magprof
import myhelm_magstar as myhmag
import rhoTcontours as rtc
from StarModCore import maghydrostar_core


class maghydrostar(maghydrostar_core):
	"""
	Magnetohydrostatic star generator.  Generates spherical WDs with 
	adiabatic temperature profiles using the Helmholtz 
	(http://cococubed.asu.edu/code_pages/eos.shtml) EOS, rigid rotation 
	spherical approximation and Gough & Tayler 66 magnetic convective 
	suppression criterion (but no pressure term).  Includes gravitational 
	and varying specific heat ratio terms of GT66 criterion.  All values 
	in CGS.

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

	Parameters (Unique to maghydrostar)
	-----------------------------------
	S_want : use a user-specified central entropy (erg/K) INSTEAD OF 
		temperature temp_c.
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
	densest : central density initial estimate for self.getstarmodel().
	omegaest : estimate of rigid rotating angular speed.  Default is False
		- code wil then use 0.75*mystar.L_want/I.
	dontintegrate : set up the problem, but don't shoot for a star.

	Returns
	-------
	mystar : maghydrostar class instance
		If star was integrated and data written, results can be found in
		mystar.data.  Further analysis can be performed with 
		mystar.getenergies,	mystar.getgradients, mystar.getconvection
        and mystar.gettimescales.

	Notes
	-----
	Additional documentation can be found for specific functions within the
	class, and in the parent class maghydrostar_core.  The default behaviour 
    of maghydrostar is to shoot for either a user-specified mass or both a 
    mass and an angular momentum.  It is possible to define an instance of 
    maghydrostar and then use integrate_star to	produce profiles of a 
    known density, central temperature/entropy, and	spin angular velocity 
    Omega.  Be warned, however, that many of the integration parameters, 
    including the value of the temperature floor, the types of 
    superadiabatic temperature gradient deviations used, and the 
	pressure and density at which to halt integration, are not automatically 
    updated.  MAKE SURE these are set when the class instance is declared!  
    See Examples, below.

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
	>>> import copy as cp
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
	>>>     out_dict["stars"].append(cp.deepcopy(mystar.data))
	"""

	def __init__(self, mass, temp_c, magprofile=False, omega=0., Lwant=0., 
				S_want=False, mintemp=1e5, composition="CO", togglecoulomb=True,
				simd_userot=True, simd_usegammavar=False, simd_usegrav=False, 
				simd_suppress=False, P_end_ratio=1e-8, ps_eostol=1e-8, 
				fakeouterpoint=False, stop_invertererr=True, 
				stop_mrat=2., stop_positivepgrad=True, stop_mindenserr=1e-10, 
				densest=False, omegaest=False, mass_tol=1e-6, L_tol=1e-6, 
				omega_crit_tol=1e-3, nreps=30, stopcount_max=5, 
				dontintegrate=False, verbose=True):

		maghydrostar_core.__init__(self, mass, temp_c, magprofile=magprofile, 
				omega=omega, Lwant=Lwant, mintemp=mintemp,
				composition=composition, togglecoulomb=togglecoulomb,
				fakeouterpoint=fakeouterpoint, stop_invertererr=stop_invertererr,
				stop_mrat=stop_mrat, stop_positivepgrad=stop_positivepgrad, 
				stop_mindenserr=stop_mindenserr, mass_tol=mass_tol, L_tol=L_tol, 
				omega_crit_tol=omega_crit_tol, nreps=nreps, 
				stopcount_max=stopcount_max, verbose=verbose)

		# Set up derivative functions
		self.simd_userot = simd_userot
		self.simd_usegammavar = simd_usegammavar
		self.simd_usegrav = simd_usegrav
		self.simd_suppress = simd_suppress
		self.nablarat_crit = False			# This should only be used for debugging!

		if self.simd_usegammavar and self.verbose:
			print "VARIABLE GAMMA TERM included in nabla deviation!"
		if self.simd_usegrav and self.verbose:
			print "GRAVITY TERM included in nabla deviation!"
		if self.simd_userot and self.verbose:
			print "ROTATION TERM included in nabla deviation!"
		if self.simd_suppress and self.verbose:
			print "WARNING - YOU'LL BE SUPPRESSING nabla DEVIATIONS IN THE FIRST MASS STEP!  I HOPE YOU HAVE A GOOD REASON!"

		# derivatives_gtsh now handles both constant deviation and user-defined magnetic profile.
		self.derivatives = self.derivatives_gtsh
		self.first_deriv = self.first_derivatives_gtsh
		if self.verbose:
			print "GT66/MM09 derivative selected!"

		if dontintegrate:
			if self.verbose:
				print "WARNING: integration disabled within maghydrostar!"
		else:
			if self.omega < 0.:
				self.omega = 0.
				self.getmaxomega(P_end_ratio=P_end_ratio, densest=densest, S_want=S_want, ps_eostol=ps_eostol)
			else:
				if Lwant:
					self.getrotatingstarmodel_2d(densest=densest, omegaest=omegaest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, damp_nrstep=0.25)
				else:
					self.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol)

		# Checks omega, just to make sure user didn't initialze a "dontintegrate" but set omega < 0
		assert self.omega >= 0.


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
		if self.nabladev:
			Bfld = np.sqrt(4.*np.pi*Gamma1*press*self.nabladev/(1. - self.nabladev))
		else:
			Bfld = self.magf.fBfld(R, mass)
		Pchi = (1./8./np.pi)*Bfld**2

		nabla_terms = {}

		if isotherm:

			hydrograd = 0.		        # Zero out hydrograd and deviation; totalgrad then will equal 0.
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
		if self.nabladev:
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
				Gamma1_est = self.getgamma_PD(dens, P, failtrig=failtrig)
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

		# If we use self.nabladev make sure it isn't negative
		if self.nabladev:
			assert self.nabladev > 0.

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


	def getgradients(self):
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


########################################### BLANK STARMOD FOR POST-PROCESSING #############################################


	def backcalculate_subloop(self):
		"""maghydrostar subloop to backcalculate.
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
			y_in = np.array([self.data["R"][i], temp_data["Pgas"][i], self.data["T"][i]])
			[dummydy, dummyBfld, temp_data["Pmag"][i], temp_data["nabla_hydro"][i], temp_data["nabla_mhdr"][i], nabla_terms] = self.derivatives(y_in, self.data["M"][i], self.omega, m_step=m_step[i], grad_full=True)

			if self.simd_usegammavar:
				self.data["dlngamdlnP"][i] = nabla_terms["dlngamdlnP"]
				self.data["nd_gamma"][i] = nabla_terms["nd_gamma"]
			if self.simd_usegrav:
				self.data["dlngdlnP"][i] = nabla_terms["dlngdlnP"]
				self.data["nd_grav"][i] = nabla_terms["nd_grav"]
			if self.simd_userot:
				self.data["nd_rot"][i] = nabla_terms["nd_rot"]

		return temp_data

