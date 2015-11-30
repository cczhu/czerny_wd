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

class endofsimmer(maghydrostar):
	"""End of simmering class, child of maghydrostar (mainly to use the derivative and Helmholtz-based functions within that class.  Initializing the class only initializes it for use; to generate models, must call getsimmermodel.

	Arguments:
	mass: mass wanted
	mintemp: isothermal temperature floor
	composition: "CO", "Mg" or "He" composition
	P_end_ratio: ratio of P/P_c at which to terminate integration
	tol: relative mass tolerance (mass - mass wanted)/mass wanted
	nreps: number of shooting attempts in getsimmermodel
	fakeouterpoint: add additional point to profile where M = mass_want to prevent interpolation attempts from failing
	stop_invertererr: stop when EOS error is reached
	stop_mrat: stop when integrated M is larger than stop_mrat*mass
	stop_positivepgrad: stop when dP/dr becomes positive
	ps_eostol: EOS tolerance when inverting
	omegatol: when using the omega iterative finder, absolute error tolerance for maximum omega
	stopcount_max: when the code detects that mass wanted may not be achievable, number of stellar integration iterations to take before testing for global extrema
	dontintegrate: set up the problem, but don't shoot for a star
	verbose: report happenings within code
	"""

	def __init__(self, mass, mintemp=1e4, composition="CO", P_end_ratio=1e-8,
					mass_tol=1e-6, nreps=100, fakeouterpoint=False, stop_invertererr=True, stop_mrat=2., 
					stop_positivepgrad=True, ps_eostol=1e-8, stopcount_max=5, dontintegrate=False, verbose=True):

		omega=0.
		omegatol = 1e-2

		self.P_end_ratio = P_end_ratio
		self.fakeouterpoint = fakeouterpoint
		self.stop_invertererr = stop_invertererr
		self.stop_mrat = stop_mrat
		self.stop_positivepgrad = stop_positivepgrad
		self.ps_eostol = ps_eostol
		self.stopcount_max = stopcount_max
		self.verbose = verbose

		magzero = magprof.magprofile(None, None, None, blankfunc=True)

		maghydrostar.__init__(self, mass, False, magzero, S_want=1e8, omega=omega, mass_tol=mass_tol, composition=composition, derivtype="sim",
							simd_usegammavar=False, simd_usegrav=False, simd_usegrav_rcut=0.0, simd_suppress=False, simd_userot=False,
							nablarat_crit=False, nrc_dens=1e30, P_end_ratio=self.P_end_ratio, densest=False, nreps=nreps,
							mintemp=mintemp, fakeouterpoint=self.fakeouterpoint, stop_invertererr=self.stop_invertererr, stop_mrat=self.stop_mrat,
							stop_positivepgrad=self.stop_positivepgrad, ps_eostol=self.ps_eostol, omegatol=omegatol,
							stopcount_max=self.stopcount_max, dontintegrate=True, verbose=self.verbose)
	

	def getsimmermodel(self, S_want, S_old, Mconv, Lconvrat=False, densest=False, out_search=False):
		"""Obtains star at end of convective simmering, where (super-)adiabatic temperature gradient is used before user defined mass shell M_conv, and user-defined entropy profile is used after.

		Arguments:
		S_want: central entropy
		S_old: previous entropy profile
		Mconv: enclosed mass of convection zone
		densest: central density estimate (false for code to guess)
		out_search: return trial densities and corresponding masses
		"""

 		if S_old:
 			self.S_old = S_old	#Store old entropy structure
 		else:
 			self.S_old = lambda x: S_want
 
 		if Mconv:
 			self.Mconv = Mconv
 		else:
 			self.Mconv = 1e100

		if Lconvrat:
			self.Lconvrat = Lconvrat
			td = rtc.timescale_data(max_axes=[1e12,1e12])
			self.eps_nuc_interp = td.getinterp2d("eps_nuc")
		else:
			self.Lconvrat = False

		if not densest:
			densest = 3.73330253e-60*self.mass_want*self.mass_want*3.	#The 3. is recently added to reduce the time for integrating massive WDs from scratch
		stepmod = 0.1*densest

		i = 1
		[Mtot, outerr_code] = self.integrate_simmer(densest, S_want, outputerr=True)
		beforedirec = int((self.mass_want - Mtot)/abs(self.mass_want - Mtot))	#1 means next shot should have larger central density, -1 means smaller)
		stepmod = float(beforedirec)*stepmod	#Set initial direction (same notation as above)
		if self.verbose:
			print "First shot: M = {0:.5e} (vs. wanted M = {1:.5e})".format(Mtot, self.mass_want)
			print "Direction is {0:d}".format(beforedirec)

		#Minor additional test
		M_arr = np.array([Mtot])
		dens_arr = np.array([densest])
		checkglobalextreme = False

		#If we incur hugemass_err the first time
		if outerr_code == "hugemass_err":
			if self.verbose:
				print "hugemass_err is huge from the first estimate.  Let's try a much lower density."
			densest = 0.1*densest
			[Mtot, outerr_code] = self.integrate_simmer(densest, S_want, outputerr=True)

		#If we incur any error (or hugemass_err again)
		if outerr_code:
			print "OUTERR_CODE {0:s}!  EXITING FUNCTION!".format(outerr_code)
			return outerr_code

		while abs(Mtot - self.mass_want) >= self.mass_tol*self.mass_want and i < self.nreps and beforedirec != 0 and not outerr_code:

			[stepmodchange, beforedirec] = self.checkdirec(beforedirec, self.mass_want - Mtot)	#For the first time, this will give stepmodchange = 1, beforedirec = 1
			stepmod *= stepmodchange
			densest += stepmod
			if densest <= 1e1:	#Central density estimate really shouldn't be lower than about 1e4 g/cc
				densest = abs(stepmod)*0.1
				stepmod = 0.1*stepmod
			if self.verbose:
				print "Old density estimate rho = {0:.5e}; new rho = {1:.5e}".format(densest - stepmod, densest)

			[Mtot, outerr_code] = self.integrate_simmer(densest, S_want, outputerr=True)
			if self.verbose:
				print "Current shot: M = {0:.5e} (vs. wanted M = {1:.5e})".format(Mtot, self.mass_want)
			M_arr = np.append(M_arr, Mtot)
			dens_arr = np.append(dens_arr, densest)

			#Check for extreme circumstances where a solution might be impossible
			if int((dens_arr[i] - dens_arr[i-1])/abs(dens_arr[i] - dens_arr[i-1])) != int((M_arr[i] - M_arr[i-1])/abs(M_arr[i] - M_arr[i-1])) and not checkglobalextreme:
				print "Density and mass don't have a positive-definite relationship in the regime you selected!"
				if i == 1:	#Check that increasing (decreasing) central density increases (decreases) mass
					print "We're at the beginning of mass finding (i = 1), so reversing direction!"
					stepmod *= -1.0
				checkglobalextreme = True
				stopcount = 0

			#If we've already activated checkglobalextreme, we should only integrate up to stopcount = stopcount_max
			if checkglobalextreme:
				stopcount += 1
				if stopcount > self.stopcount_max:		#If we've integrated one times too many after checkglobalextreme = True
					outerr_code = self.chk_global_extrema(M_arr, Mtot - self.mass_want, self.mass_want)

			i += 1

		if beforedirec == 0:
			print "ERROR! checkdirec is printing out 0!"
		if i == self.nreps:
			print "WARNING, maximum number of shooting attempts {0:d} reached!".format(i)

		#If we incur any error (or hugemass_err again)
		if outerr_code:
			print "OUTERR_CODE {0:s}!  EXITING FUNCTION!".format(outerr_code)
			return outerr_code

		if self.verbose:
			print "Final shot!"
		Mtot = self.integrate_simmer(densest, S_want, recordstar=True)

		if abs((Mtot - self.mass_want)/self.mass_want) > self.mass_tol:
			print "ERROR!!!! (M_total - mass_want)/mass_want = {0:.5e}".format((Mtot - self.mass_want)/self.mass_want)
			print "THIS IS BIGGER THAN YOUR TOLERANCE!  CHECK YOUR ICS!"
		else:
			print "(M_total - mass_want)/mass_want = {0:.5e}".format((Mtot - self.mass_want)/self.mass_want)

		if out_search:
			return [M_arr, dens_arr]


	def integrate_simmer(self, dens_c, S_c, recordstar=False, outputerr=False):

		stepsize = self.mass_want*0.01*self.mass_tol

		if self.ps_eostol > 1e-8 and self.verbose:
			print "WARNING: self.ps_eostol is now {0:.3e}".format(self.ps_eostol)

		#Print out the M = 0 step
		M = 0
		R = 0
		entropy = S_c
		dens = dens_c
		[P, temp] = self.getpress_rhoS(dens, entropy)
		Pend = P*self.P_end_ratio
		if recordstar:	#auto-clears any previous data dict
			self.data = {"M": [M],
			"R": [R],
			"Pgas": [P],
			"rho": [dens],
			"T": [temp],
			"Sgas": [entropy],
			"Pmag": [],	#Pmag, B will be populated in the next step (done in case B(r = 0) = infty)
			"B": [],
			"nabla_hydro": [],
			"nabla_mhd": [],
			"nabla_terms": [],
			"superad_grad": [],
			"superad_dev": []}

		if self.Lconvrat:
			self.Lnuc_endsimmer = [0]
			self.Lconv_endsimmer = [0]
			self.eps_nuc_endsimmer = [self.eps_nuc_interp(dens, temp)]

		#Take one step forward (we know the central values), assuming the central density does not change significantly
		M = stepsize
		if M > self.Mconv:
			R, P, temp, Bfld, Pmag, hydrograd, maggrad, nabla_terms, superad_grad, superad_dev = self.first_deriv_simmer_oS(dens, M, P, temp)
		else:
			R, P, temp, Bfld, Pmag, hydrograd, maggrad, nabla_terms, superad_grad, superad_dev = self.first_deriv_simmer(dens, M, P, temp)

		if recordstar:
			self.data["M"].append(M)
			self.data["R"].append(R)
			self.data["Pgas"].append(P)
#			self.data["rho"].append(dens)
#			self.data["T"].append(temp)
#			self.data["Sgas"].append(entropy)
			self.data["Pmag"].append(Pmag)	#Pmag, B populated for the first time here
			self.data["B"].append(Bfld)
			self.data["nabla_hydro"].append(hydrograd)
			self.data["nabla_mhd"].append(maggrad)
			self.data["nabla_terms"].append(nabla_terms)
			self.data["superad_grad"].append(superad_grad)
			self.data["superad_dev"].append(superad_dev)

		errtrig = [-10]		#Errortrig, used to stop integrator when Helmholtz NR-method fails to properly invert the EOS (usually due to errors being slightly beyond tolerance levels)

		#Continue stepping using scipy.integrate.odeint
		while P > Pend:

			if M > self.Mconv:
				entropy = self.getS_old(M)
				[dens, temp] = self.getdens_PS_est(P, entropy, dens_est=dens, temp_est=temp, eostol=self.ps_eostol)

				if self.Lconvrat:
					self.advanceLconv_endsimmer(dens, temp, M, stepsize)

				y_in = np.array([R, P])
				[dy, Bfld, Pmag, hydrograd, maggrad, nabla_terms, superad_grad, superad_dev] = self.derivatives_simmer_oS(y_in, M, dens_est=dens, temp_est=temp, grad_full=True)
				stepsize = self.stepcoeff*min(abs(y_in/(dy+1e-35)))

				if recordstar:
					self.data["rho"].append(dens)
					self.data["T"].append(temp)
					self.data["Sgas"].append(entropy)
					self.data["Pmag"].append(Pmag)
					self.data["B"].append(Bfld)
					self.data["nabla_hydro"].append(hydrograd)
					self.data["nabla_mhd"].append(maggrad)
					self.data["nabla_terms"].append(nabla_terms)
					self.data["superad_grad"].append(superad_grad)
					self.data["superad_dev"].append(superad_dev)

				R, P = scipyinteg.odeint(self.derivatives_simmer_oS, y_in, np.array([M,M+stepsize]), args=(errtrig, dens, temp), h0=stepsize*0.01, hmax = stepsize)[1,:]

			else:
				[dens, entropy] = self.getdens_PT(P, temp)

				if self.Lconvrat:
					self.advanceLconv_endsimmer(dens, temp, M, stepsize)

				#Adaptive stepsize
				y_in = np.array([R, P, temp])
				[dy, Bfld, Pmag, hydrograd, maggrad, nabla_terms, superad_grad, superad_dev] = self.derivatives_simmer(y_in, M, m_step=stepsize, grad_full=True)	#The only way to "self-consistently" print out B is in the derivative, where it could be modified to force B^2/Pgas to be below some critical value
				stepsize = self.stepcoeff*min(abs(y_in/(dy+1e-35)))

				if recordstar:
					self.data["rho"].append(dens)
					self.data["T"].append(temp)
					self.data["Sgas"].append(entropy)
					self.data["Pmag"].append(Pmag)
					self.data["B"].append(Bfld)
					self.data["nabla_hydro"].append(hydrograd)
					self.data["nabla_mhd"].append(maggrad)
					self.data["nabla_terms"].append(nabla_terms)
					self.data["superad_grad"].append(superad_grad)
					self.data["superad_dev"].append(superad_dev)

				R, P, temp = scipyinteg.odeint(self.derivatives_simmer, y_in, np.array([M,M+stepsize]), args=(errtrig, 1e-8, stepsize), h0=stepsize*0.01, hmax = stepsize, mxstep=1000)[1,:]


			M += stepsize

			if temp < 1.5e3:
				lordjesu = theword

			if recordstar:
				self.data["M"].append(M)
				self.data["R"].append(R)
				self.data["Pgas"].append(P)

			outerr_code = False
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

		#Step outward one last time
		if M > self.Mconv:
			entropy = self.getS_old(M)
			[dens, temp] = self.getdens_PS_est(P, entropy, dens_est=dens, temp_est=temp, eostol=self.ps_eostol)
			if self.Lconvrat:
				self.advanceLconv_endsimmer(dens, temp, M, stepsize)
			y_in = np.array([R, P])
			[dy, Bfld, Pmag, hydrograd, maggrad, nabla_terms, superad_grad, superad_dev] = self.derivatives_simmer_oS(y_in, M, dens_est=dens, temp_est=temp, grad_full=True)
		else:
			[dens, entropy] = self.getdens_PT(P, temp)
			if self.Lconvrat:
				self.advanceLconv_endsimmer(dens, temp, M, stepsize)
			y_in = np.array([R, P, temp])
			[dy, Bfld, Pmag, hydrograd, maggrad, nabla_terms, superad_grad, superad_dev] = self.derivatives_simmer(y_in, M, m_step=stepsize, grad_full=True)

		#Generate a fake data point where M is exactly mass_want.  Useful for interpolators.
		if recordstar:
			self.data["rho"].append(dens)
			self.data["T"].append(temp)
			self.data["Sgas"].append(entropy)
			self.data["Pmag"].append(Pmag)
			self.data["B"].append(Bfld)
			self.data["nabla_hydro"].append(hydrograd)
			self.data["nabla_mhd"].append(maggrad) 
			self.data["nabla_terms"].append(nabla_terms)
			self.data["superad_grad"].append(superad_grad)
			self.data["superad_dev"].append(superad_dev)

			if self.fakeouterpoint:
				if M > self.Mconv:
					stepsize = self.stepcoeff*min(abs(y_in/(dy+1e-35)))
					R, P = scipyinteg.odeint(self.derivatives_simmer_oS, y_in, np.array([M,M+stepsize]), args=(errtrig, dens, temp), h0=stepsize*0.01, hmax = stepsize)
				else:
					stepsize = self.stepcoeff*min(abs(y_in/(dy+1e-35)))
					R, P, temp = scipyinteg.odeint(self.derivatives_simmer, y_in, np.array([M,M+stepsize]), args=(errtrig, 1e-8, stepsize), h0=stepsize*0.01, hmax = stepsize)[1,:]
				M += stepsize
				self.data["M"].append(max(self.mass_want, M))
				self.data["R"].append(R)
				self.data["Pgas"].append(0.)
				#Gradients one last time
				if M > self.Mconv:
					entropy = self.getS_old(M)
					[dens, temp] = self.getdens_PS_est(P, entropy, dens_est=dens, temp_est=temp, eostol=self.ps_eostol)
					if self.Lconvrat:
						self.advanceLconv_endsimmer(dens, temp, M, stepsize)
					y_in = np.array([R, P])
					[dy, Bfld, Pmag, hydrograd, maggrad, nabla_terms, superad_grad, superad_dev] = self.derivatives_simmer_oS(y_in, M, dens_est=dens, temp_est=temp, grad_full=True)
				else:
					[dens, entropy] = self.getdens_PT(P, temp)
					if self.Lconvrat:
						self.advanceLconv_endsimmer(dens, temp, M, stepsize)
					y_in = np.array([R, P, temp])
					[dy, Bfld, Pmag, hydrograd, maggrad, nabla_terms, superad_grad, superad_dev] = self.derivatives_simmer(y_in, M, m_step=stepsize, grad_full=True)
				self.data["rho"].append(0.)
				self.data["T"].append(0.)
				self.data["Sgas"].append(0.)
				self.data["Pmag"].append(Pmag)
				self.data["B"].append(Bfld)
				self.data["nabla_hydro"].append(hydrograd)
				self.data["nabla_mhd"].append(maggrad)
				self.data["nabla_terms"].append(nabla_terms)
				self.data["superad_grad"].append(superad_grad)
				self.data["superad_dev"].append(superad_dev)

			self.unpack_nabla_terms()
			for item in self.data.keys():
				self.data[item] = np.array(self.data[item])
				if self.Lconvrat:
					self.exportLconv()

		if outputerr:
			return [M, outerr_code]
		else:
			return M


	def getS_old(self, M):
		#If decoupled from central convection zone, revert to old entropy
		return float(self.S_old(M))


	def advanceLconv_endsimmer(self, rho, T, M, Mstep):
		self.eps_nuc_endsimmer.append(self.eps_nuc_interp(rho, T))
		newLnuc = self.Lnuc_endsimmer[-1] + 0.5*Mstep*(self.eps_nuc_endsimmer[-1] + self.eps_nuc_endsimmer[-2])
		self.Lnuc_endsimmer.append(newLnuc)
		self.Lconv_endsimmer.append(newLnuc*self.Lconvrat(M))


	def exportLconv(self):
		self.eps_nuc_endsimmer = np.array(self.eps_nuc_endsimmer)
		self.Lnuc_endsimmer = np.array(self.Lnuc_endsimmer)
		self.Lconv_endsimmer = np.array(self.Lconv_endsimmer)
		

	def first_deriv_simmer(self, dens, M, Pc, Tc):

		R, P, temp, Bfld, Pchi, hydrograd, maggrad, nabla_terms = self.first_deriv(dens, M, Pc, Tc)

		superad_grad = 0.
		superad_dev = 0.

		return [R, P, temp, Bfld, P, hydrograd, maggrad, nabla_terms, superad_grad, superad_dev]


	def derivatives_simmer(self, y, mass, failtrig=[-100], ps_eostol=1e-8, m_step=1e29, grad_full=False):

		if grad_full:
			[dydx, Bfld, Pchi, hydrograd, maggrad, nabla_terms] = self.derivatives(y, mass, failtrig=failtrig, ps_eostol=ps_eostol, m_step=m_step, grad_full=grad_full)
		else:
			dydx = self.derivatives(y, mass, failtrig=failtrig, ps_eostol=ps_eostol, m_step=m_step)

		if self.Lconvrat and y[2] > self.mintemp:
			[dens, entropy] = self.getdens_PT(y[1], y[2], failtrig=failtrig)
			[adgradred_dumm, hydrograd, nu_dumm, alpha_dumm, delta, Gamma1, cP, cPhydro_dumm] = self.geteosgradients(dens, y[2], 0.)
			H_Pagrav = y[1]/dens

			superad_grad = (self.Lconv_endsimmer[-1]/(np.pi*y[0]**2*dens*cP*y[2]))**(2./3.)*(2./(delta*H_Pagrav))**(1./3.)
			superad_dev = y[2]/y[1]*superad_grad*dydx[1]

			if y[2] <= 33.*self.mintemp:
				superad_dev = self.mintemp_func(y[2])*superad_dev

			dydx[2] += superad_dev
		else:
			superad_grad = 0.
			superad_dev = 0.

		if grad_full:
			if self.Lconvrat:
				hydrograd += superad_grad
				maggrad += superad_grad
			return [dydx, Bfld, Pchi, hydrograd, maggrad, nabla_terms, superad_grad, superad_dev]
		else:
			return dydx


	def first_deriv_simmer_oS(self, dens, M, P, Tc):

		R = (3.*M/(4.*np.pi*dens))**(1./3.)
		moddens = 4./3.*np.pi*dens
		P = P - (3.*self.grav/(8.*np.pi)*moddens**(4./3.) - 0.25/np.pi*self.omega**2*moddens**(1./3.))*M**(2./3.)	#This is integrated out assuming constant density

		[adgradred_dumm, hydrograd, nu_dumm, alpha_dumm, delta, Gamma1, cP_dumm, cPhydro_dumm] = self.geteosgradients(dens, Tc, 0.)

#		superad_grad = 0.
#		superad_dev = 0.

		return [R, P, 0.0, 0.0, 0.0, hydrograd, 0.0, {}, 0., 0.]


	def derivatives_simmer_oS(self, y, mass, failtrig=[-100], dens_est=1e6, temp_est=1e7, grad_full=False):
		R = y[0]
		press = y[1]
		[dens, temp] = self.getdens_PS_est(press, self.getS_old(mass), failtrig=failtrig, dens_est=dens_est, temp_est=temp_est, eostol=self.ps_eostol)
		dydx = np.zeros(2)
		dydx[0] = 1./(4.*np.pi*R**2.*dens)
		dydx[1] = -self.grav*mass/(4.*np.pi*R**4.) + 1./(6.*np.pi)*self.omega**2/R

		#if you previously calculated a temperature gradient that included superadiabaticity, this gradient is already included in S_old

#		if self.Lconvrat:
#			[adgradred_dumm, hydrograd, nu_dumm, alpha_dumm, delta, Gamma1, cP, cPhydro_dumm] = self.geteosgradients(dens, temp, 0.)
#			H_Pagrav = press/dens

#			superad_grad = (self.Lconv_endsimmer[-1]/(np.pi*R**2*dens*cP*temp))**(2./3.)*(2./(delta*H_Pagrav))**(1./3.)
#			superad_dev = temp/press*superad_grad*dydx[1]
#			dydx[2] += superad_dev
#		else:
#			superad_grad = 0.
#			superad_dev = 0.

		if grad_full:
#			if self.Lconvrat:
#				hydrograd += superad_grad
#			else:
			[adgradred_dumm, hydrograd, nu_dumm, alpha_dumm, delta, Gamma1, cP_dumm, cPhydro_dumm] = self.geteosgradients(dens, temp, 0.)
			return [dydx, 0.0, 0.0, hydrograd, hydrograd, {}, 0., 0.]
		else:
			return dydx

