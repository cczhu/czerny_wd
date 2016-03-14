import numpy as np
import pylab
from scipy.interpolate import interp1d
import scipy.integrate as scipyinteg
import os, sys
import myhelm_magstar as myhmag
import rhoTcontours as rtc
from StarModSteve import mhs_steve
import magprofile_mass as magprof
import Sprofile_mass as sprof


class mhs_snd(mhs_steve):
	"""
    Testbed for new derivatives
	"""

	def __init__(self, mass, S_want, magprofile=False, omega=0., L_want=0., 
				temp_c=False, mintemp=1e5, composition="CO", togglecoulomb=True,
				S_old=False, P_end_ratio=1e-8, ps_eostol=1e-8, 
				fakeouterpoint=False, stop_invertererr=True, 
				stop_mrat=2., stop_positivepgrad=True, stop_mindenserr=1e-10, 
				densest=False, omegaest=False, mass_tol=1e-6, L_tol=1e-6, 
				omega_crit_tol=1e-3, nreps=30, stopcount_max=5, verbose=True):

		mhs_steve.__init__(self, mass, S_want, magprofile=magprofile, omega=omega, L_want=L_want, 
				temp_c=temp_c, mintemp=mintemp, composition=composition, togglecoulomb=togglecoulomb,
				S_old=S_old, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, 
				fakeouterpoint=fakeouterpoint, stop_invertererr=stop_invertererr, 
				stop_mrat=stop_mrat, stop_positivepgrad=stop_positivepgrad, stop_mindenserr=stop_mindenserr, 
				densest=densest, omegaest=omegaest, mass_tol=mass_tol, L_tol=L_tol, 
				omega_crit_tol=omega_crit_tol, nreps=nreps, stopcount_max=stopcount_max, 
				dontintegrate=True, verbose=verbose)

		self.derivatives = self.derivatives_steve
		self.first_deriv = self.first_derivatives_steve

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



######################################## DERIVATIVES #######################################


	def derivatives_steve(self, y, mass, omega, Fconv, failtrig=[-100], ps_eostol=1e-8, m_step=1e29, isotherm=False, grad_full=False):
		"""
		Derivative that uses Stevenson 1979's formulation for superadiabatic convective deviations.
		"""
		R = y[0]
		press = y[1]
		temp = y[2]

		[dens, entropy] = self.getdens_PT(press, temp, failtrig=failtrig)

		# Take mag pressure Pchi = 0 for calculating hydro coefficients
		[adgradred, hydrograd, nu, alpha, delta, Gamma1, cP, cPhydro, c_s] = self.geteosgradients(dens, temp, 0., failtrig=failtrig)

		dydx = np.zeros(3)
		dydx[0] = 1./(4.*np.pi*R**2.*dens)
		dptotaldm = -self.grav*mass/(4.*np.pi*R**4.) + 1./(6.*np.pi)*omega**2/R
		dydx[1] = dptotaldm 	#- Pchi_grad*dydx[0]

		if self.nabladev:
			Bfld = np.sqrt(8.*np.pi*press*self.nabladev)
		else:
			Bfld = self.magf.fBfld(R, mass)
		Pchi = (1./8./np.pi)*Bfld**2

		if isotherm:

			hydrograd = 0.												# Zero out hydrograd and deviation; totalgrad then will equal 0.
			nabla_terms = {"v_conv_st": 0., "c_s_st": c_s, "nd": 0.}	# Populate deviations as zero

		else:

			nabla_terms = {"c_s_st": c_s}

			agrav = self.grav*mass/R**2.										# grav modulus = GM_enc/r^2
			H_P = min(-press*dydx[0]/dptotaldm, (press/self.grav/dens**2)**0.5)	# H_P = min(-P/(dP/dR), sqrt(P/G\rho^2)) (Eggleton 71 approx.)

			nabla_terms["v_conv_st"] = self.vc_coeff*(delta*agrav*H_P/cP/temp*Fconv/dens)**(1./3.)
			nabla_terms["nd"] = self.nab_coeff*(1./delta)*nabla_terms["v_conv_st"]**2/(agrav*H_P)
			#nabla_terms["nd"] = (1./delta)*(nabla_terms["v_conv_st"]/nabla_terms["c_s_st"])**2		# c_s = sqrt(g*H_P) (Stevenson 79 sentence below Eqn. 37)

			if omega > 0.:
				rossby = nabla_terms["v_conv_st"]/(2.*omega*H_P)					# v_0/(2*omega*H_P)
				nabrat = (1. + (6./25./np.pi**2)**(4./5.)*rossby**(-8./5.))**0.5
				nabla_terms["v_conv_st"] = nabla_terms["v_conv_st"]*nabrat**(-1./4.)
				nabla_terms["nd"] = nabla_terms["nd"]*nabrat						# obtain v from v_0
			elif Bfld > 0.:
				alfrat = Bfld**2/(4.*np.pi*dens*nabla_terms["v_conv_st"]**2)
				nabla_terms["v_conv_st"] = nabla_terms["v_conv_st"]*(1. + 0.538*alfrat**(13./8.))**(-4./13.)
				nabla_terms["nd"] = nabla_terms["nd"]*(1. + 0.18*alfrat**(6./5.))**(5./6.)

		if self.nablarat_crit and (abs(nabla_terms["nd"])/hydrograd > self.nablarat_crit):
			raise AssertionError("ERROR: Hit critical nabla!  Code is now designed to throw an error so you can jump to the point of error.")

		totalgrad = hydrograd + nabla_terms["nd"]
		dydx[2] = temp/press*totalgrad*dydx[1]

		if grad_full:
			return [dydx, Bfld, Pchi, hydrograd, totalgrad, nabla_terms]
		else:
			return dydx


	def first_derivatives_steve(self, dens, M, Pc, Tc, omega, failtrig=[-100]):
		"""
		First step to take for self.derivatives_steve().   R, P, temp are at r = R,
		while hydrograd, totalgrad, nabla_terms are derivative and magnetic field values
		for r = 0.
		"""

		R = (3.*M/(4.*np.pi*dens))**(1./3.)
		moddens = 4./3.*np.pi*dens
		P = Pc - (3.*self.grav/(8.*np.pi)*moddens**(4./3.) - 0.25/np.pi*omega**2*moddens**(1./3.))*M**(2./3.)	# This is integrated out assuming constant density and magnetic field strength

		if self.nabladev:
			Bfld = np.sqrt(8.*np.pi*Pc*self.nabladev)
		else:
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

		# Since this code is both derivative and integrator, R, P and temp are integrated values at r = R, while
		# Bfld...nabla_terms are derivative and associated values at r = 0.
		return [R, P, temp, Bfld, Pchi, hydrograd, totalgrad, nabla_terms, dy_est, isotherm]

