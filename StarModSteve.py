import numpy as np
import pylab
from scipy.interpolate import interp1d
import scipy.integrate as scipyinteg
import os, sys
import myhelm_magstar as myhmag
import rhoTcontours as rtc
from StarMod import maghydrostar
import magprofile_mass as magprof

################EndSimmer CLASS###############################

class mhs_steve(maghydrostar):
	"""
	Magnetohydrostatic star generator.  Generates spherical WDs with 
	adiabatic temperature profiles using the Helmholtz 
	(http://cococubed.asu.edu/code_pages/eos.shtml) EOS, rigid rotation 
	spherical approximation based on Stevenson 1979 and Barker et al. 2014.  
	All values in CGS.

	Parameters
	----------
	mass : wanted mass (g)
	S_want : user-specified central entropy (erg/K)
	magprofile : magnetic profile object.  Defaults to false, meaning no 
		magnetic field.  If derivtype="sim", a magf object containing the 
		field should be passed.  If derivtype="simcd", magprofile should 
		equal delta = B^2/(B^2 + 4pi*Gamma1*Pgas).
	omega : rigid rotation angular velocity (rad/s).  Defaults to 0 (non-
		rotating).  If < 0, attempts to estimate break-up omega with 
		self.getomegamax(), if >= 0, uses user defined value.
	Lwant : wanted angular momentum.
	temp_c : wanted central temperature (K).  If nonzero, will use this instead
		of central entropy S_want.
	mintemp : temperature floor (K), effectively switches from adiabatic 
		to isothermal profile if reached.
	stop_mindenserr : density floor, below which integration is halted.
		Default is set to 1e-10 to prevent it from ever being reached.  
		Helmholtz sometimes has trouble below this 1e-8; try adjusting this
		value to eliminate inverter errors.
	simd_userot : use Solberg-Hoiland deviation from adiabatic temperature 
		gradient.
	simd_usegammavar : use gamma = c_P/c_V index magnetic deviation from 
		adiabatic temperature gradient.
	simd_usegrav : use gravity magnetic devation from adiabatic temperature 
		gradient.
	densest : central density initial estimate for self.getstarmodel().
	omegaest : estimate of rigid rotating angular speed.  Default is False
		- code wil then use 0.75*mystar.L_want/I.
	mass_tol : fractional tolerance between mass wanted and mass produced 
		by self.getstarmodel()
	L_tol : fractional tolerance between L wanted and L produced 
		by self.getrotatingstarmodel()
	omega_crit_tol : when using the self.getomegamax() iterative finder, 
		absolute error tolerance for maximum omega.
	verbose : report happenings within code.
	dontintegrate: don't perform any integration

	Returns
	-------
	mystar : mhs_steve class instance
		If star was integrated and data written, results can be found in
		mystar.data.  Further analysis can be performed with 
		mystar.getenergies,	mystar.getgradients and mystar.getconvection.
	"""

	def __init__(self, mass, S_want, magprofile=False, omega=0., Lwant=0., 
				temp_c=False, mintemp=1e4, stop_mindenserr=1e-10,
				simd_userot=True, simd_usegammavar=False, simd_usegrav=False,
				densest=False, omegaest=False, mass_tol=1e-6, L_tol=1e-6, 
				omega_crit_tol=1e-3, verbose=True, dontintegrate=False,
				**kwargs):

		td = rtc.timescale_data(max_axes=[1e12,1e12])
		self.eps_nuc_interp = td.getinterp2d("eps_nuc")

		# maghydrostar class initialization: DONTINTEGRATE MUST BE SET
		# TO TRUE FOR CURRENT CODE TO WORK!
		maghydrostar.__init__(self, mass, temp_c, magprofile=magprofile, 
				omega=omega, Lwant=Lwant, S_want=S_want, mintemp=mintemp, 
				derivtype="sim", stop_mindenserr=stop_mindenserr, 
				simd_userot=simd_userot, simd_usegammavar=simd_usegammavar, 
				simd_usegrav=simd_usegrav, densest=densest, omegaest=omegaest, 
				mass_tol=mass_tol, L_tol=L_tol, omega_crit_tol=omega_crit_tol,
				dontintegrate=True, verbose=verbose, **kwargs)

		self.derivatives = self.derivatives_steve
		self.first_deriv = self.first_derivatives_steve

		if dontintegrate:
			if self.verbose:
				print "WARNING: integration disabled within maghydrostar!"
		else:
			if omega < 0.:
				self.getmaxomega(densest=densest, S_want=S_want)
			else:
				if Lwant:
					self.getrotatingstarmodel_2d(densest=densest, omegaest=omegaest, S_want=S_want, damp_nrstep=0.25)
				else:
					self.getstarmodel(densest=densest, S_want=S_want)


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

		Notes
		----------
		In Python, operator overloading is done using "__" before the method name.
		We actually want the maghydrostar functions to be overridden, so we don't
		do this.  See:
		http://stackoverflow.com/questions/12764995/python-overriding-an-inherited-class-method
		https://docs.python.org/2/reference/expressions.html#atom-identifiers
		"""

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
			"nabla_terms": []}

		errtrig = [0]		# Errortrig to stop integrator when Helmholtz NR-method fails to properly invert the EOS (usually due to errors slightly beyond tolerance)

		# Load convection luminosity data
		self.fconv_data = {"Lnuc": [0],
					"Lconv": [0],
					"Fconv": [0],
					"eps_nuc": [self.eps_nuc_interp(dens, temp)]}

		# Take one step forward (we know the central values, and we assume dP_chi/dr = 0), assuming the central density does not change significantly
		# first_deriv also returns the starting value of isotherm
		M = stepsize
		R, P, temp, Bfld, Pmag, hydrograd, totalgrad, nabla_terms, isotherm = self.first_deriv(
																				dens, M, Pc, temp, omega, failtrig=errtrig)
		[dens, entropy] = self.getdens_PT(P, temp, failtrig=errtrig)
		self.advanceFconv(dens, R, temp, M, stepsize)

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
			self.data["Pmag"].append(Pmag)	# Pmag, B, etc, 
			self.data["B"].append(Bfld)
			self.data["nabla_hydro"].append(hydrograd)
			self.data["nabla_mhdr"].append(totalgrad)
			self.data["nabla_terms"].append(nabla_terms)

		# Continue stepping using scipy.integrate.odeint
		while P > Pend:

			# Adaptive stepsize
			y_in = np.array([R, P, temp])
			# The only way to "self-consistently" print out B is in the derivative, where it could be modified to force B^2/Pgas to be below some critical value
			[dy, Bfld, Pmag, hydrograd, totalgrad, nabla_terms] = self.derivatives(y_in, M, omega, self.fconv_data["Fconv"][-1],
																		ps_eostol=ps_eostol, m_step=stepsize, isotherm=isotherm, grad_full=True)
			stepsize = self.stepcoeff*min(abs(y_in/(dy+1e-35)))

			if recordstar:
				self.data["Pmag"].append(Pmag)
				self.data["B"].append(Bfld)
				self.data["nabla_hydro"].append(hydrograd)
				self.data["nabla_mhdr"].append(totalgrad)
				self.data["nabla_terms"].append(nabla_terms)

			R, P, temp = scipyinteg.odeint(self.derivatives, y_in, np.array([M,M+stepsize]), 
												args=(omega, self.fconv_data["Fconv"][-1], errtrig, ps_eostol, stepsize, isotherm), 
												h0=stepsize*0.01, hmax = stepsize, mxstep=1000)[1,:]

			if temp <= self.mintemp and not isotherm:
				R, P, temp, M = self.connect_isotherm(y_in, M, stepsize, omega, Pend, errtrig, ps_eostol, isotherm)
				isotherm = True
			else:
				M += stepsize

			[dens, entropy] = self.getdens_PT(P, temp, failtrig=errtrig)

			self.advanceFconv(dens, R, temp, M, stepsize)

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
		[dy, Bfld, Pmag, hydrograd, totalgrad, nabla_terms] = self.derivatives(y_in, M, omega, self.fconv_data["Fconv"][-1], ps_eostol=ps_eostol, m_step=stepsize, isotherm=isotherm, grad_full=True)

		# Generate a fake data point where M is exactly mass_want.  Useful for setting initial conditions for 3D WD simulations.
		if recordstar:
			self.data["Pmag"].append(Pmag)
			self.data["B"].append(Bfld)
			self.data["nabla_hydro"].append(hydrograd)
			self.data["nabla_mhdr"].append(totalgrad) 
			self.data["nabla_terms"].append(nabla_terms) 

			if self.fakeouterpoint:
				stepsize = self.stepcoeff*min(abs(y_in/(dy+1e-30)))
				y_out = scipyinteg.odeint(self.derivatives, y_in, np.array([M,M+stepsize]), args=(omega, self.fconv_data["Fconv"][-1], errtrig, ps_eostol, stepsize, isotherm), h0=stepsize*0.01, hmax = stepsize)[1,:]
				self.advanceFconv(0., y_out[0], 0., M+stepsize, Mstep)
				self.data["M"].append(max(self.mass_want, M+stepsize))
				self.data["R"].append(y_out[0])
				self.data["Pgas"].append(0.)
				self.data["rho"].append(0.)
				self.data["T"].append(0.)
				self.data["Sgas"].append(0.)
				y_in = y_out	# Gradients one last time
				[dy, Bfld, Pmag, hydrograd, totalgrad, nabla_terms] = self.derivatives(y_in, M, omega, self.fconv_data["Fconv"][-1], ps_eostol=ps_eostol, m_step=stepsize, isotherm=isotherm, grad_full=True)
				self.data["Pmag"].append(Pmag)
				self.data["B"].append(Bfld)
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


	def connect_isotherm(self, y_in, M, stepsize, omega, Pend, errtrig, ps_eostol, isotherm, iter_max=1000, subtol=1e-8):
		"""
		Stevenson 1979-compatible loop to achieve self.mintemp within tolerance self.mintemp_reltol.
		"""
		substep = 0.1*stepsize			# we'll use constant stepsizes
		ys_in = np.array(y_in)			# arrays are passed by reference, not by value!
		P = ys_in[1]
		sM = M
		i = 0
		while i < iter_max and P > Pend and abs(substep) > subtol*stepsize:			# Integrate forward toward self.mintemp
			R, P, temp = scipyinteg.odeint(self.derivatives, ys_in, np.array([sM,sM+substep]), 
												args=(omega, self.fconv_data["Fconv"][-1], errtrig, ps_eostol, substep, 
												isotherm), h0=substep*0.01, hmax = substep, mxstep=1000)[1,:]

			sM += substep

			# If we're within the temperature tolerance
			if abs(temp - self.mintemp)/self.mintemp < self.mintemp_reltol:
				break

			# If we've overshot
			if temp < self.mintemp:
				R, P, temp = np.array(ys_in)	# reset R, P, temp to start of step
				sM -= substep					# reverse mass step
				substep = 0.1*substep			# repeat last integration step with greater accuracy

			# If we've done neither, set up next integration
			ys_in = np.array([R, P, temp])
				
		return [R, P, temp, sM]


	def advanceFconv(self, dens, R, T, M, Mstep, togglecoulomb=True):
		"""
		Integrates Fconv, step by step using trapezoidal rule
		(https://en.wikipedia.org/wiki/Trapezoidal_rule), where 
		F = 0.5*SUM(dx*(y_i+1 + y_i)).  Here, L = 0.5*SUM(dM*(eps[i+1] + eps[i])),
		and Eth = 0.5*SUM(dM*(eth[i+1] + eth[i]))
		"""
		# Use current eps_nuc and previous to integrate to current Lnuc
		self.fconv_data["eps_nuc"].append(self.eps_nuc_interp(dens, T))	# Calculate CURRENT eps_nuc
		self.fconv_data["Lnuc"].append(self.fconv_data["Lnuc"][-1] + 0.5*Mstep*(self.fconv_data["eps_nuc"][-1] + self.fconv_data["eps_nuc"][-2]))
		# Use eth to integrate current Eth
#		[new_eint, new_edeg, dumm_c_s] = self.gethelmeos_energies(dens, T, togglecoulomb=togglecoulomb)
#		self.fconv_data["eth"].append(new_eint - new_edeg)				# Calculate CURRENT eth
#		self.fconv_data["Eth"].append(self.fconv_data["Eth"][-1] + 0.5*Mstep*(self.fconv_data["eth"][-1] + self.fconv_data["eth"][-2]))

		self.fconv_data["Lconv"].append(self.fconv_data["Lnuc"][-1])			#TEMPORARY

		self.fconv_data["Fconv"].append(self.fconv_data["Lconv"][-1]/(4.*np.pi*R**2))		# Can't integrate this independently, of course.


	def unpack_nabla_terms(self):
		len_nabla = len(self.data["nabla_terms"])
		for item in self.data["nabla_terms"][0].keys():
			self.data[item] = np.zeros(len_nabla)
		for i in range(len_nabla):
			for item in self.data["nabla_terms"][0].keys():
				self.data[item][i] = self.data["nabla_terms"][i][item]
		del self.data["nabla_terms"]	#delete nabla_terms to remove duplicate copies


	def unpack_Fconv(self):
#		len_fconv = len(self.fconv_data["Lnuc"])
#		for item in self.fconv_data.keys():
#			self.data[item] = np.array(self.fconv_data[item])
		for item in self.fconv_data.keys():
			self.fconv_data[item] = np.array(self.fconv_data[item])


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
			nabla_terms = {"v_conv": 0., "c_s": c_s, "nd": 0.}	# Populate deviations as zero

		else:

			nabla_terms = {"c_s": c_s}

			agrav_eff = -dptotaldm/dydx[0]/dens		# g_eff = -dP/dr/rho
			H_P = min(-press*dydx[0]/dptotaldm, (press/self.grav/dens**2)**0.5)	# H_P = min(-P/(dP/dR), sqrt(P/G\rho^2)) (Eggleton 71 approx.)
			nabla_terms["c_s"] = (agrav_eff*delta*H_P/8.)**0.5					# c_s = sqrt(g*delta*H_P/8) (Comparing Stevenson 79 to K&W MLT)

			nabla_terms["v_conv"] = (0.25*delta*agrav_eff*H_P/cP/temp*Fconv/dens)**(1./3.)

			if omega == 0.:
				nabla_terms["nd"] = (nabla_terms["v_conv"]/nabla_terms["c_s"])**2
			else:
				nabla_terms["nd"] = (nabla_terms["v_conv"]/nabla_terms["c_s"])*(2*H_P*omega/nabla_terms["c_s"])

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

		nabla_terms = {"v_conv": 0., "c_s": c_s, "nd": 0.}

		totalgrad = hydrograd + nabla_terms["nd"]	# Curently nabla_terms["nd"] = 0

		temp = Tc + Tc/Pc*totalgrad*(P - Pc)		

		# If we hit the temperature floor, artificially force temp to remain at it, and set integration to isothermal
		if temp <= self.mintemp:
			temp = self.mintemp
			isotherm = True
		else:
			isotherm = False

		return [R, P, temp, Bfld, Pchi, hydrograd, totalgrad, nabla_terms, isotherm]
