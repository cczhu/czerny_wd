import numpy as np
import scipy.optimize as sciopt

################### ROTATING SOLUTION ########################

def s79_rotating_low_est(rossby):
	"""Eqn 43 for R << 1"""
	eps_low = (6./25./np.pi**2)**0.4*rossby**-(4./5.)		# 0.23R_0^-0.8
	v_low = eps_low**-0.25
	return [eps_low, v_low]


def s79_rotating_high_est(rossby):
	"""Eqn 42 for R >> 1"""
	eps_high = 1. + (4./25./np.pi**2)*rossby**-2		# 1 + 1/(62R_0^2)
	v_high = 1. - (242.)**-1*rossby**-2					# Eqn 42
	return [eps_high, v_high]


def s79_rotating_soln(rossby_lim=[1e-5, 1e5]):

	f_fit = lambda x, rossby: x + (6./25./np.pi**2)/rossby**2 - x**2.5

	rossby_range = 10**np.linspace(np.log10(rossby_lim[0]), np.log10(rossby_lim[1]), num=1000)

	eps_out = np.zeros(len(rossby_range))
	for i in range(len(rossby_range)):
		if rossby_range[i] < 0:
			eps_est, v_est = s79_rotating_low_est(rossby_range[i])
		else:
			eps_est, v_est = s79_rotating_high_est(rossby_range[i])
		eps_out[i] = sciopt.fsolve(f_fit, eps_est, args=(rossby_range[i]))

	v_out = eps_out**-0.25

	return [rossby_range, eps_out, v_out]


def check_s79_rotating(rossby_lim = [1e-5, 1e5]):
	""" Checks rotating exact solution (S79 Eqns. 40 - 43) vs. Eqns. 5.41 and 5.42
	in Thesis.
	"""

	[rossby_range, eps_out, v_out] = s79_rotating_soln(rossby_lim=rossby_lim)

	[eps_low, v_low] = s79_rotating_low_est(rossby_range)
	[eps_high, v_high] = s79_rotating_high_est(rossby_range)

	eps_est = (1. + (6./25./np.pi**2)**(4./5.)*rossby_range**(-8./5.))**0.5
	v_est = eps_est**(-1./4.)

	plt.figure()
	plt.title("$\Delta_\mathrm{rot}$")
	plt.loglog(rossby_range, eps_out, 'r-', label="exact")
	plt.loglog(rossby_range, eps_high, 'g--', label="R_0 >> 1 approx")
	plt.loglog(rossby_range, eps_low, 'b--', label="R_0 << 1 approx")
	plt.loglog(rossby_range, eps_est, 'c--', lw=2, label="estimate")
	plt.ylim(1e-1, 1e5); plt.xlim(10**-6.5, 10**3.5)
	plt.xlabel("Rossby Number $R_0$"); plt.ylabel(r"$\Delta/\Delta_0$")
	plt.xlabel("Rossby Number $R_0$")

	plt.figure()
	plt.title("$v_\mathrm{rot}$")
	plt.loglog(rossby_range, v_out, 'r-', label="exact")
	plt.loglog(rossby_range, v_high, 'g--', label="R_0 >> 1 approx")
	plt.loglog(rossby_range, v_low, 'b--', label="R_0 << 1 approx")
	plt.loglog(rossby_range, v_est, 'c--', lw=2, label="estimate")
	plt.ylim(10**-1., 10**0.5); plt.xlim(10**-6.5, 10**3.5)
	plt.xlabel("Rossby Number $R_0$"); plt.ylabel(r"$v/v_0$")
	plt.legend(loc=2)

	print "Max fractional diff of estimated and exact nabla_rot/nabla_ad = {0:.3e}".format(max(abs(eps_est - eps_out)/eps_out))
	print "Max fractional diff of estimated and exact v_rot/v_ad = {0:.3e}".format(max(abs(v_est - v_out)/v_out))

################### MAGNETIC SOLUTION (S79 Eqns. 40 - 43) ########################


def s79_innerprod_sq(kappa_vec):
	return (kappa_vec[0]**2 + kappa_vec[1]**2 + kappa_vec[2]**2)


def s79_mag_kvec(gab, w_bsq):
	""" returns reduced vec-k (i.e. k times l_0/pi) """
	gamma = w_bsq/gab
	k_y = (3. + (9. - 16.*gamma + 16.*gamma**2)**0.5)/4./(1. - gamma)
	k_y = k_y**0.5
	if type(gab) == np.ndarray:
		kappa_vec = np.zeros([3, len(gab)])
		kappa_vec[2] = 1.*np.ones(len(gab))		# k[2] = pi/l_0
		kappa_vec[1] = np.array(k_y)
	else:
		kappa_vec = np.array([0, k_y, 1.])
	return kappa_vec


def s79_mag_sigma(gab, w_bsq):
	""" returns sigma """
	kappa_vec = s79_mag_kvec(gab, w_bsq)
	return (gab*kappa_vec[1]**2/s79_innerprod_sq(kappa_vec) - w_bsq)**0.5


def s79_mag_v(gab, w_bsq):
	kappa_vec = s79_mag_kvec(gab, w_bsq)
	sigma = s79_mag_sigma(gab, w_bsq)
	return sigma/s79_innerprod_sq(kappa_vec)**0.5


def s79_mag_fred(gab, w_bsq):
	""" returns reduced flux (i.e. without the rho*cP/alpha/g, and where k^2 has been divided by pi^2/l_0^2) """
	kappa_vec = s79_mag_kvec(gab, w_bsq)
	sigma = s79_mag_sigma(gab, w_bsq)
	return sigma**3/s79_innerprod_sq(kappa_vec)*(1. + w_bsq/sigma**2)


def s79_mag_soln(wb_lim = [1e-5, 1e0]):
	w_bsq_range = 10**np.linspace(wb_lim[0], wb_lim[1], 1000.) 

	alfrat = np.array([])
	v = np.array([])
	eps = np.array([])

	for i in range(len(w_bsq_range)):

		gab = 10.**np.linspace(np.log10(w_bsq_range[i]) + 0.0001, np.log10(w_bsq_range[i]) + 0.001, 10)
		gab = np.concatenate([gab, 10.**np.linspace(np.log10(w_bsq_range[i]) + 0.001, np.log10(w_bsq_range[i]) + 4., 5000)])		# wb**2 < gab < 1e3*wb**2

		flux = s79_mag_fred(gab, w_bsq_range[i])
		v_m = s79_mag_v(gab, w_bsq_range[i])
		gab_0 = (flux*(3./2.)*(3./5.)**-2.5)**(2./3.)
		v_0 = (6./25.*gab_0)**0.5

		v = np.concatenate([v, v_m/v_0])
		eps = np.concatenate([eps, gab/gab_0])
		alfrat = np.concatenate([alfrat, w_bsq_range[i]/v_0**2])

	sortargs = np.argsort(alfrat)
	alfrat_range = alfrat[sortargs]
	v_out = v[sortargs]
	eps_out = eps[sortargs]
	
	return [alfrat_range, eps_out, v_out]


def s79_mag_low_est(alfrat):
	"""Eqn 46 for A << 1"""
	eps_low = 1. + 2.*alfrat/15.
	v_low = 1. - 11.*alfrat/75.
	return [eps_low, v_low]


def s79_mag_high_est(alfrat):
	"""Eqn 47 for A >> 1"""
	eps_high = 0.24*alfrat
	v_high = 1.21*alfrat**-0.5	# Error in s79 - should be 1.21A^-1/2, not 0.92A^-1/2; makes no qualitative difference though, since scaling is identical
	return [eps_high, v_high]


def check_s79_magnetic(wb_lim = [1e-5, 1e0]):
	""" Checks rotating exact solution (S79 Eqns. 40 - 43) vs. Eqns. 5.41 and 5.42
	in Thesis.
	"""

	[alfrat_range, eps_out, v_out] = s79_mag_soln(wb_lim=wb_lim)

	[eps_low, v_low] = s79_mag_low_est(alfrat_range)
	[eps_high, v_high] = s79_mag_high_est(alfrat_range)

	eps_est = (1. + 0.18*alfrat_range**(6./5.))**(5./6.)
	v_est = (1. + 0.538*alfrat_range**(13./8.))**(-4./13.)

	plt.figure()
	plt.title("$\Delta_\mathrm{mag}$")
	plt.loglog(alfrat_range, eps_out, 'r-', label="exact")
	plt.loglog(alfrat_range, eps_high, 'g--', label="A >> 1 approx")
	plt.loglog(alfrat_range, eps_low, 'b--', label="A << 1 approx")
	plt.loglog(alfrat_range, eps_est, 'c--', lw=2, label="estimate")
	plt.ylim(10**-0.5,1e4); plt.xlim(10**-5, 10**5)
	plt.xlabel("Alfven Ratio $A$"); plt.ylabel(r"$\Delta/\Delta_0$")
	plt.xlabel("Alfven Ratio $A$")

	plt.figure()
	plt.title("$v_\mathrm{mag}$")
	plt.loglog(alfrat_range, v_out, 'r-', label="exact")
	plt.loglog(alfrat_range, v_high, 'g--', label="A >> 1 approx")
	plt.loglog(alfrat_range, v_low, 'b--', label="A << 1 approx")
	plt.loglog(alfrat_range, v_est, 'c--', lw=2, label="estimate")
	plt.ylim(1e-3,1e1); plt.xlim(10**-5, 10**5)
	plt.xlabel("Alfven Ratio $A$"); plt.ylabel(r"$v/v_0$")
	plt.legend(loc=3)

	print "Max fractional diff of estimated and exact nabla_mag/nabla_ad = {0:.3e}".format(max(abs(eps_est - eps_out)/eps_out))
	print "Max fractional diff of estimated and exact v_mag/v_ad = {0:.3e}".format(max(abs(v_est - v_out)/v_out))
