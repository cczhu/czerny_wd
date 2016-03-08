import StarMod as Star
import magprofile_mass as magprof
import numpy as np
import copy
import os, sys

def obtain_model(mystar, i, r_in, verbose=False):
	"""Used by make_runaway to calculate models with increasing entropy S_want.
	"""

	S_want = r_in["S_arr"][i]	# Central entropy wanted for this step.

	# Obtain initial density and omega estimates from previous (the coefficients are purely empirical)
	densest = 0.8*mystar.data["rho"][0]
	if r_in.has_key("L_original"):
		omegaest = 0.9*mystar.omega

	# Empirically density estimates below 1e6 are less accurate, so we need more error tolerance
	if densest < r_in["lowerr_tol"]:
		ps_eostol = 100.*r_in["ps_eostol"]
		P_end_ratio = 100.*r_in["P_end_ratio"]
		print "Running at reduced eostol = {0:.3e}; P_end_ratio = {1:.3e}".format(ps_eostol, P_end_ratio)
	elif densest < r_in["mederr_tol"]:
		ps_eostol = 10.*r_in["ps_eostol"]
		P_end_ratio = 10.*r_in["P_end_ratio"]
		print "Running at reduced eostol = {0:.3e}; P_end_ratio = {1:.3e}".format(ps_eostol, P_end_ratio)
	else:
		ps_eostol = r_in["ps_eostol"]
		P_end_ratio = r_in["P_end_ratio"]

	# Right now I'll define these here, but if we want to reduce the stepcoeffs
	deltastepcoeff_rho = 0.1
	deltastepcoeff_omega = 0.1
	damp_nrstep = 0.25

#	if r_in.has_key("L_original"):
#		outerr_code = mystar.getrotatingstarmodel(densest=densest, omegaest=omegaest, S_want=S_want, P_end_ratio=r_in["P_end_ratio"], ps_eostol=r_in["ps_eostol"], 
#													damp_nrstep=damp_nrstep, deltastepcoeff=deltastepcoeff_omega, interior_dscoeff=deltastepcoeff_rho, 
#													omega_warn=1.)
#		if outerr_code:
#			print "-----------HACK - OUTERR_CODE OUTPUTTED BY OVERLOOP, TRYING AGAIN WITH 0.5*OMEGA ---------------"
#			outerr_code = mystar.getrotatingstarmodel(densest=densest, omegaest=0.5*omegaest, S_want=S_want, P_end_ratio=r_in["P_end_ratio"], ps_eostol=r_in["ps_eostol"], 
#													damp_nrstep=damp_nrstep, deltastepcoeff=deltastepcoeff_omega, interior_dscoeff=deltastepcoeff_rho, 
#													omega_warn=1.)
	if r_in.has_key("L_original"):
		outerr_code = mystar.getrotatingstarmodel_2d(densest=densest, omegaest=omegaest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, 
													damp_nrstep=damp_nrstep, deltastepcoeff=deltastepcoeff_omega, omega_warn=1.)
		if outerr_code:
			print "-----------HACK - OUTERR_CODE OUTPUTTED BY OVERLOOP, TRYING AGAIN WITH A LOWER OMEGA AND TIGHTER STEPPING---------------"
			outerr_code = mystar.getrotatingstarmodel_2d(densest=densest, omegaest=omegaest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, 
													damp_nrstep=damp_nrstep, deltastepcoeff=deltastepcoeff_omega, omega_warn=1.)
	else:	# For now, keep P_end_ratio=r_in["P_end_ratio"], ps_eostol=r_in["ps_eostol"]
		outerr_code = mystar.getstarmodel(densest=densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, 
										deltastepcoeff=deltastepcoeff_rho)

		if outerr_code:
			print "-----------HACK - STAR OUTPUT FAILURE, TRYING AGAIN WITH A 5% INCREASE IN DENSITY ESTIMATE--------------------"
			outerr_code = mystar.getstarmodel(densest=1.05*densest, S_want=S_want, P_end_ratio=P_end_ratio, ps_eostol=ps_eostol, 
											deltastepcoeff=deltastepcoeff_rho)

	return outerr_code


def make_runaway(starmass=1.2*1.9891e33, mymag=False, omega=0., omega_run_rat=0.8, S_arr=10**np.arange(7.5,8.2,0.25), 
					mintemp=1e5, tog_coul=True, mass_tol=1e-3, P_end_ratio=1e-8, densest=False, stop_mindenserr=1e-10, 
					L_tol=1e-2, keepstars=False, simd_userot=True, simd_usegammavar=True, simd_usegrav=True, simd_suppress=False, 
					omega_crit_tol=1e-2, omega_warn=10., om_err_tol=[1e6, 1e7], verbose=True):
	"""Obtains runaway of a star of some given mass, magnetic field, and rotation.  Outputs an object (usually several hundred megs large) that includes run inputs as "run_inputs", as well as all stellar output curves (hence its size) under "stars".

	Arguments:
	starmass: wanted mass
	mymag: magnetic profile object.  Defaults to false, meaning no magnetic field.
	omega: rigid rotation angular velocity.  Defaults to 0 (non-rotating).  If < 0, attempts to estimate break-up omega with self.getomegamax(), if >= 0, uses user defined value.
	S_arr: list of central entropy values in the runaway track
	mintemp: temperature floor, effectively switches from adiabatic to isothermal profile if reached
	mass_tol: fractional tolerance between mass wanted and mass produced by self.getstarmodel()
	P_end_ratio: ratio of P/P_c at which to terminate stellar integration
	densest: central density initial estimate for self.getstarmodel()
	L_tol: conservation of angular momentum error tolerance
	simd_userot: use Solberg-Hoiland deviation from adiabatic temperature gradient
	simd_usegammavar: use gamma = c_P/c_V index magnetic deviation from adiabatic temperature gradient
	simd_usegrav: use gravity magnetic devation from adiabatic temperature gradient
	simd_suppress: suppresses deviations from adiabaticity in first step of integration
	omega_crit_tol: when using mystar.getomegamax(), absolute error tolerance for maximum omega
	omega_warn: stop integration within mystar.getrotatingstarmodel() if self.omega approaches omega_warn*omega_crit estimate.  Defaults to 10 to prevent premature stoppage.
	verbose: report happenings within code
	"""

	# Save a few details about the magnetic field (though we'll need better records for the actual paper)
	try:
		mymagsave = np.array([float(mymag.fBfld_r(0)), float(mymag.fBfld_r(5e8))])
	except:
		if verbose:
			print "Magnetic field not found!  Hopefully this is what you wanted."
		mymagsave = False

	r_in = {"mass": starmass, 
			"magprofile": mymagsave, 
			"omega": omega, 
			"omega_run_rat": omega_run_rat, 
			"S_arr": S_arr, 
			"mintemp": mintemp,  
			"composition": "CO",
			"tog_coul": tog_coul,
			"simd_userot": simd_userot, 
			"simd_usegammavar": simd_usegammavar, 
			"simd_usegrav": simd_usegrav, 
			"simd_suppress": simd_suppress, 
			"P_end_ratio": P_end_ratio, 
			"ps_eostol": 1e-8, 
			"fakeouterpoint": False, 
			"stop_invertererr": True, 
			"stop_mrat": 2., 
			"stop_positivepgrad": True, 
			"stop_mindenserr": stop_mindenserr,
			"densest": densest, 
			"mass_tol": mass_tol,
			"L_tol": L_tol, 
			"omega_crit_tol": omega_crit_tol, 
			"omega_warn": omega_warn,
			"lowerr_tol": om_err_tol[0],
			"mederr_tol": om_err_tol[1]}

	if (omega != 0) or mymag:
		print "*************You want to make an MHD/rotating star; let's first try making a stationary pure hydro star!************"
		mymagzero = magprof.magprofile(None, None, None, None, blankfunc=True)
		hstar = Star.maghydrostar(r_in["mass"], max(5e6,mintemp), magprofile=mymagzero, omega=0., S_want=False, mintemp=r_in["mintemp"], composition=r_in["composition"], togglecoulomb=r_in["tog_coul"], simd_userot=r_in["simd_userot"], simd_usegammavar=r_in["simd_usegammavar"], simd_usegrav=r_in["simd_usegrav"], simd_suppress=r_in["simd_suppress"], P_end_ratio=r_in["P_end_ratio"], ps_eostol=r_in["ps_eostol"], fakeouterpoint=r_in["fakeouterpoint"], stop_invertererr=r_in["stop_invertererr"], stop_mindenserr=r_in["stop_mindenserr"], stop_mrat=r_in["stop_mrat"], stop_positivepgrad=r_in["stop_positivepgrad"], densest=r_in["densest"], mass_tol=r_in["mass_tol"], L_tol=r_in["L_tol"], omega_crit_tol=r_in["omega_crit_tol"], nreps=100, verbose=verbose)
		densest=0.9*hstar.data["rho"][0]

	print "*************Okay, let's make a low-temperature (possibly MHD/rotating) star************"
	mystar = Star.maghydrostar(r_in["mass"], max(5e6,mintemp), magprofile=mymag, omega=r_in["omega"], S_want=False, 	#Rest after this is identical to function call above
				mintemp=r_in["mintemp"], composition=r_in["composition"], togglecoulomb=r_in["tog_coul"], simd_userot=r_in["simd_userot"], simd_usegammavar=r_in["simd_usegammavar"], simd_usegrav=r_in["simd_usegrav"], simd_suppress=r_in["simd_suppress"], P_end_ratio=r_in["P_end_ratio"], ps_eostol=r_in["ps_eostol"], fakeouterpoint=r_in["fakeouterpoint"], stop_invertererr=r_in["stop_invertererr"], stop_mindenserr=r_in["stop_mindenserr"], stop_mrat=r_in["stop_mrat"], stop_positivepgrad=r_in["stop_positivepgrad"], densest=r_in["densest"], mass_tol=r_in["mass_tol"], L_tol=r_in["L_tol"], omega_crit_tol=r_in["omega_crit_tol"], nreps=100, verbose=verbose)

	# If we use a high-temperature mintemp, remove any S_arr
	if S_arr[0] < mystar.data["Sgas"][0]:
		i_S_arr = min(np.where(S_arr > mystar.data["Sgas"][0])[0])
		S_arr = S_arr[i_S_arr:]
		r_in["S_arr"] = S_arr

	if r_in["omega"] < 0:
		print "FOUND critical Omega = {0:.3e}!  We'll use {1:.3e} of this value for the runaway.".format(mystar.omega, r_in["omega_run_rat"])
		r_in["omega_crit_foundinfirststep"] = mystar.omega
		mystar.omega *= r_in["omega_run_rat"]

		mystar.getstarmodel(densest=0.9*mystar.data["rho"][0], P_end_ratio=r_in["P_end_ratio"], ps_eostol=r_in["ps_eostol"])

	if r_in["omega"]:
		if mystar.omega > mystar.getcritrot(max(mystar.data["M"]), mystar.data["R"][-1]):
			print "WARNING: exceeding estimated critical rotation!  Consider restarting this run."
		r_in["L_original"] = mystar.getmomentofinertia(mystar.data["R"], mystar.data["rho"])[-1]*mystar.omega
		mystar.L_want = r_in["L_original"]			# Store initial angular momentum for future use.

	out_dict = {"temp_c": np.zeros(len(r_in["S_arr"])+1),
		"dens_c": np.zeros(len(r_in["S_arr"])+1),
		"omega": np.zeros(len(r_in["S_arr"])+1),
		"B_c": np.zeros(len(r_in["S_arr"])+1),
		"S_c": np.zeros(len(r_in["S_arr"])+1),
		"R": np.zeros(len(r_in["S_arr"])+1),
		"stars": []}

	if "R_nuc" not in mystar.data.keys():	# Obtain timescale info if it's not already printed.
		mystar.gettimescales()

	out_dict["run_inputs"] = r_in
	if r_in["omega"] < 0:
		out_dict["omega_crit"] = r_in["omega_crit_foundinfirststep"]
	out_dict["S_c"][0] = mystar.data["Sgas"][0]
	out_dict["temp_c"][0] = mystar.data["T"][0]
	out_dict["dens_c"][0] = mystar.data["rho"][0]
	out_dict["omega"][0] = mystar.omega
	out_dict["B_c"][0] = np.mean(mystar.data["B"][:10])
	out_dict["R"][0] = mystar.data["R"][-1]
	if keepstars:
		out_dict["stars"].append(copy.deepcopy(mystar))
	else:
		out_dict["stars"].append(copy.deepcopy(mystar.data))

	for i in range(len(r_in["S_arr"])):

		print "*************Star #{0:d}, entropy = {1:.3f}************".format(i, r_in["S_arr"][i])

		outerr_code = obtain_model(mystar, i, r_in, verbose=verbose)

		if outerr_code:
			if outerr_code == "nrepsnotconverged_err":
				print "===== RUNAWAY.PY REPORTS OUTERR: ", outerr_code, "but code hacked, so keep going! ====="
			else:
				print "===== RUNAWAY.PY REPORTS OUTERR: ", outerr_code, "so will stop model making! ====="
				break

		if "R_nuc" not in mystar.data.keys():	# Obtain timescale info if it's not already printed.
			mystar.gettimescales()
		out_dict["S_c"][i+1] = mystar.data["Sgas"][0]
		out_dict["temp_c"][i+1] = mystar.data["T"][0]
		out_dict["dens_c"][i+1] = mystar.data["rho"][0]
		out_dict["omega"][i+1] = mystar.omega
		out_dict["B_c"][i+1] = np.mean(mystar.data["B"][:10])
		out_dict["R"][i+1] = mystar.data["R"][-1]
		if keepstars:
			out_dict["stars"].append(copy.deepcopy(mystar))
		else:
			out_dict["stars"].append(copy.deepcopy(mystar.data))

	return out_dict
