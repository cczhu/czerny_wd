########## TEST CASES ###############
# Suite of test cases to run whenever major
# changes in the code are made.  Convective
# suppression cannot be tested against other
# codes, only for self-consistency in the
# solution!  Importing these test cases
# into analysis scripts can be useful when
# checking production runs.

import numpy as np
import scipy.interpolate as sci_interp
import scipy.integrate as sci_integ
from scipy.interpolate import UnivariateSpline
import StarMod as Star
import runaway as rw
import os, sys
import copy
import rhoTcontours as rtc
import matplotlib.pyplot as plt

sys.path.append("/home/cczhu/Runaway")
import runaway_prod_anal_func as rpaf

##################### SUPPORT FUNCTIONS ####################

def remove_unfinished_runs(od):
	i_cut = max((od["dens_c"] > 0).nonzero()[0])
	odnew = {}
	for item in od.keys():
		if item == 'run_inputs':
			odnew[item] = copy.deepcopy(od[item])
		else:
			try:
				odnew[item] = copy.deepcopy(od[item][:i_cut+1])
			except:
				odnew[item] = copy.deepcopy(od[item])
	return odnew


def tc_recast_array(X, Y, Xr):
	"""Checks the ratio of two profiles ([X1, Y1] and [X2, Y2]) that have different lengths.

	Arguments:
	X/Y: x and y values
	Xr: rebinned x values
	"""
	Ysout = sci_interp.UnivariateSpline(X, Y, k=1, s=0, ext=3)
	Yr = Ysout(Xr)
	return Yr


def tc_get_ratio_uneven_arrays(X1, Y1, X2, Y2, returntype=None):
	"""Checks the ratio of two stellar profiles ([X1, Y1] and [X2, Y2]) that may have different lengths.
	Assumes X starts at 0.

	Parameters 
	----------
	X1/Y1 : x and y values of profile 1
	X2/Y2 : x and y values of profile 2
	"""
	if max(X1) > max(X2):
		X_use = X2
		Ysout = UnivariateSpline(X1, Y1, k=1, s=0, ext=3)
		Y1 = Ysout(X_use)
	else:
		X_use = X1
		Ysout = UnivariateSpline(X2, Y2, k=1, s=0, ext=3)
		Y2 = Ysout(X_use)
	if returntype == "extrema":
		return [X_use, Y1/Y2, max(Y1/Y2), min(Y1/Y2)]
	elif returntype == "err":
		return [X_use, abs(Y1 - Y2)/Y2]
	else:
		return [X_use, Y1/Y2]


#def tc_get_ratio_uneven_arrays_OLD(X1, Y1, X2, Y2, returntype=None):
#	"""Checks the ratio of two stellar profiles ([X1, Y1] and [X2, Y2]) that may have different lengths.

#	Arguments:
#	X1/Y1: x and y values of profile 1
#	X2/Y2: x and y values of profile 2
#	"""
#	if max(X1) > max(X2):
#		X_spline = X1
#		Y_spline = Y1
#		X_use = X2
#		Y_use = Y2
#	else:
#		X_spline = X2
#		Y_spline = Y2
#		X_use = X1
#		Y_use = Y1
#	Ysout = UnivariateSpline(X_spline, Y_spline, k=1, s=0, ext=3)
#	Y_num = Ysout(X_use)
#	if returntype == "extrema":
#		return [X_use, Y_num/Y_use, max(Y_num/Y_use), min(Y_num/Y_use)]
#	elif returntype == "err":
#		return [X_use, abs(Y_num - Y_use)/Y_use]
#	else:
#		return [X_use, Y_num/Y_use]


def tc_checkmodeltol(od):
	"""Checks if mass and angular momentum are self-consistent with mass tolerance.  Magnetic flux conservation should be checked too, of course, but that is not formally bounded in the runaway code, so is checked separately under runaway_prod_anal_func.py.
	"""
	Msun = 1.9891e33
	print "User-defined mass relative tolerance: {0:e}".format(od["run_inputs"]["mass_tol"])
	masses = np.zeros(len(od["stars"]))
	M_want = od["run_inputs"]["mass"]
	for i in range(len(od["stars"])):
		masses[i] = od["stars"][i]["M"][-1]
		print "M_{0:d} = {1:e} Msun, relative err {2:e}".format(i, masses[i]/Msun, abs(masses[i] - M_want)/M_want)
	print "Violators, if any, are: ", np.arange(len(od["stars"]), dtype=int)[abs(masses - od["run_inputs"]["mass"])/od["run_inputs"]["mass"] > od["run_inputs"]["mass_tol"]]
	tester = Star.maghydrostar.blankstar(od["stars"][0])
	if od["run_inputs"]["omega"] > 0:
		Ltot = np.zeros(len(od["stars"]))
		L_original = od["run_inputs"]["L_original"]
		for i in range(len(od["stars"])):
			Ltot[i] = tester.getmomentofinertia(od["stars"][i]["R"], od["stars"][i]["rho"])[-1]*od["omega"][i]
			print "L_{0:d} = {1:e} g cm^2/s, relative err {2:e}".format(i, Ltot[i], abs(Ltot[i] - L_original)/L_original)
		print "Violators, if any, are: ", np.arange(len(od["stars"]), dtype=int)[abs(Ltot - L_original)/L_original > od["run_inputs"]["L_tol"]]

#def checkmodelfull(out_dict):
#	print "Checking self-consistency of runaway track..."
#	for i in range(len(out_dict["stars"])):
#		current_mass = max(out_dict["stars"][i]["M"])
#		current_dens_c = out_dict["dens_c"][i]
#		current_T_c = out_dict["temp_c"][i]
#		current_omega = out_dict["omega"][i]
#		if current_dens_c < 3e6:
#			ps_eostol=1e-6
#		else:
#			ps_eostol=1e-8
#		mystar= Star.hydrostar(current_mass, current_T_c, startype="isentrop", omega=current_omega, verbose=False, tol=static_tol, nreps=500, dontintegrate=True)
#		[M, outerr] = mystar.integrate_star(current_dens_c, verbose=False, P_end_ratio=1e-8, recordstar=True, stop_mrat=2., fakeouterpoint=False, stop_errtrig=True, stop_positivepgrad=True, outputerr=True, ps_eostol=ps_eostol)
#		L_current = mystar.getmomentofinertia(mystar.data["R"], mystar.data["rho"])*mystar.omega
#		print "Star {0:d}: M = {1:.3f}, L = {2:.3e} (omega = {3:.3e}), rho_c = {4:.3e}, T_c = {5:.3e}".format(i, 
#			mean(mystar.data["M"][-10:])/1.9891e33, L_current, mystar.omega, mystar.data["rho"][0], mystar.data["T"][0])
#		if outerr:
#			print outerr

def tc_compare_two_models(odod, odnd, tol=1e-3, m_encr=[0.0,0.9], verbose=True):
	"""Compares two runaway track data dicts (odod and odnd) to determine if they describe the same star to within some tolerance.  All deviations considered are relative and scaled to the value of odn.
	"""

	critical_profile_vals = ['nabla_mhdr', 'nabla_hydro', 'B', 'Sgas', 'R', 'c_s', 'vnuc', 'T', 'rho', 'Pgas', 'H_Preduced']

	ism_o = min(np.where(odod["M"] > m_encr[0]*odod["M"][-1])[0])
	ism_n = min(np.where(odnd["M"] > m_encr[0]*odnd["M"][-1])[0])
	ilg_o = max(np.where(odod["M"] < m_encr[1]*odod["M"][-1])[0]) + 1
	ilg_n = max(np.where(odnd["M"] < m_encr[1]*odnd["M"][-1])[0]) + 1

	sameflag = True

	for item in critical_profile_vals:
		y_out = tc_recast_array(odnd["M"][ism_n:ilg_n], odnd[item][ism_n:ilg_n], odod["M"][ism_o:ilg_o])
		frac_err = abs(y_out - odod[item][ism_o:ilg_o])/odod[item][ism_o:ilg_o]
		frac_err[np.isnan(frac_err)] = np.nanmean(frac_err)
		if max(frac_err) > tol:
			sameflag = False
			print "\t key = {0:s}, mean(frac_err) = {1:.3e}, max(frac_err) = {2:.3e}".format(item, np.mean(frac_err), max(frac_err))

	if sameflag and verbose:
		print "Stars are identical to within tolerance!"


def tc_compare_two_tracks(odo, odn, tol=1e-3, m_encr=[0.0,0.9]):
	"""Compares two runaway tracks (odo and odn) to determine if they describe the same runaway to within some tolerance.  All deviations considered are relative and scaled to the value of odn.
	"""

	# Check run inputs
	print "================ CHECKING RUN_INPUTS ================"
	mutual_run_inputs = list(set(odo["run_inputs"].keys()) & set(odn["run_inputs"].keys()))
	for curr_key in mutual_run_inputs:
		if curr_key == "magprofile" and type(odo["run_inputs"][curr_key]) == np.ndarray:
			if (odo["run_inputs"][curr_key][0] != odn["run_inputs"][curr_key][0]) or odo["run_inputs"][curr_key][1] != odn["run_inputs"][curr_key][1]:
				print curr_key, "=", odo["run_inputs"][curr_key], "in odo and =", odn["run_inputs"][curr_key]," in odn"
		else:
			if odo["run_inputs"][curr_key] != odn["run_inputs"][curr_key]:
				print curr_key, "=", odo["run_inputs"][curr_key], "in odo and =", odn["run_inputs"][curr_key]," in odn"

	# Single value comparisons
	print "================ CHECKING SINGLE VALUES ================"
	critical_values = ['B_c', 'temp_c', 'S_c', 'R', 'dens_c', 'omega']
	for i in range(len(odo["B_c"])):
		print "Profile {0:d}".format(i)
		for item in critical_values:
			if abs(odo[item][i]) < 1e-30:
				divisor = 1e-30
			else:
				divisor = odo[item][i]
			comparator = abs(odo[item][i] - odn[item][i])/divisor
			if comparator > tol:
				print "\t key = {0:s}, odo = {1:.3e}, odn = {2:.3e}, frac_err = {3:.3e}".format(item, odo[item][i], odn[item][i], comparator)
		comparator = abs(odo["stars"][i]["R_nuc"] - odn["stars"][i]["R_nuc"])/odo["stars"][i]["R_nuc"]
		if comparator > tol:
			print "\t key = {0:s}, odo = {1:.3e}, odn = {2:.3e}, frac_err = {3:.3e}".format("R_nuc", odo["stars"][i]["R_nuc"], odn["stars"][i]["R_nuc"], comparator)
		comparator = abs(odo["stars"][i]["Lnuc"][-1] - odn["stars"][i]["Lnuc"][-1])/odo["stars"][i]["Lnuc"][-1]
		if comparator > tol:
			print "\t key = {0:s}, odo = {1:.3e}, odn = {2:.3e}, frac_err = {3:.3e}".format("R_nuc", odo["stars"][i]["Lnuc"][-1], odn["stars"][i]["Lnuc"][-1], comparator)
		
	
	# Profiles
	print "================ CHECKING PROFILE VALUES ================"
	for i in range(len(odo["stars"])):
		odoi = odo["stars"][i]
		odni = odn["stars"][i]
		print "Profile {0:d}".format(i)
		tc_compare_two_models(odoi, odni, tol=tol, m_encr=m_encr, verbose=False)


############## CHECK AGAINST MARTEN'S CODE #################


def tc_get_marten_WD(dens_c, temp_c):
	""" Obtain WD profile from Marten's HelmholtzWD code.
	"""
	os.system("./martenwdmaker/martenwd {0:.6e} {1:.6e} {2:.6e} {3:.6e} > martentemp.txt".format(dens_c, temp_c, 0., 1e5))

	f = open("martentemp.txt", 'r')
	data2 = np.loadtxt(f, usecols=(0,1,2,3))

	out = {}
	out["R"] = data2[:,0]
	out["dens"] = data2[:,1]
	out["M"] = data2[:,2]
	out["temp"] = data2[:,3]
	
	f.close()
	os.system("rm martentemp.txt")

	return out


def tc_check_vs_marten_loop_element(ods, tol=1e-2, cutoff_mass=0.9):
	"""Checks data instance against Marten's code.
	"""

	marten = tc_get_marten_WD(ods["rho"][0], ods["T"][0])
	if len(marten["R"]) < len(ods["R"]):
		rho_default = marten["dens"]
		rho_recast = tc_recast_array(ods["R"], ods["rho"], marten["R"])
	else:
		rho_default = ods["rho"]
		rho_recast = tc_recast_array(marten["R"], marten["dens"] , ods["R"])
	rho_dev = abs(rho_recast - rho_default)/rho_recast
	cutoff = min(ods["rho"][ods["M"] < cutoff_mass*max(ods["M"])])
	args_want = rho_default > cutoff
	if max(rho_dev[args_want]) > tol:
		raise AssertionError("ERROR: maximum drho/rho = {0:.3e} > {1:.3e}".format(max(rho_dev[args_want]), tol))
	print "maximum drho/rho = {0:.3e} < {1:.3e}".format(max(rho_dev[args_want]), tol)


def tc_check_vs_marten(od, tol=1e-3):
	"""Loop for tc_check_vs_marten to act on output data structure
	"""

	for i in range(len(od["stars"])):
		tc_check_vs_marten_loop_element(od["stars"][i], tol=tol)


######### CHECK SUPERADIABATICITY SELF-CONSISTENCY #########

def recalc_temp_derivative(dc, vc_coeff=1., nab_coeff=1.):
	"""Recalculates dT/dm of stellar profile, to check if they're accurate.
	Doesn't include s_old(m) formulation or temperature floor, so it will NOT
	give exact answers after some critical point. 
	"""

	i = 0
	for dcd in dc["stars"]:
		H_P = -dcd["Pgas"]/dcd["dy"][:,1]*dcd["dy"][:,0]
		H_Pred = (dcd["Pgas"]/(6.67e-8*dcd["rho"]**2))**0.5
		args = H_P/H_Pred >= 1.; H_P[args] = H_Pred[args]
		agrav = dcd["M"]*6.67e-8/dcd["R"]**2
		dcd["vnuc0_test"] = vc_coeff*(dcd["delta"]*agrav*H_P/dcd["cP"]/dcd["T"]*dcd["Lnuc"]/4./np.pi/dcd["R"]**2/dcd["rho"])**(1./3.)
		dcd["nab0_test"] = nab_coeff*dcd["vnuc0_test"]**2/agrav/dcd["delta"]/H_P
		if dc["omega"][i] > 0:
			rossby = dcd["vnuc0_test"]/(2.*dc["omega"][i]*H_P)
			if vc_coeff < 0.99:		# for steve_cal
				rossby = rossby/3.
			epsr = (1. + (6./25./np.pi**2)**(4./5.)*rossby**(-8./5.))**0.5
			vconvr = epsr**-0.25
		elif dc["B_c"][i] > 0:
			alfrat = dcd["B"]**2/(4.*np.pi*dcd["rho"]*dcd["vnuc0_test"]**2)
			epsr = (1. + 0.18*alfrat**(6./5.))**(5./6.)
			vconvr = (1. + 0.538*alfrat**(13./8.))**(-4./13.)
		else:
			epsr = 1.
			vconvr = 1.
		dcd["nab_test"] = epsr*dcd["nab0_test"]
		dcd["vconv_test"] = vconvr*dcd["vnuc0_test"]
		totalgrad = dcd["nabla_ad"] + dcd["nab_test"]
		dcd["dTdm_test"] = dcd["T"]/dcd["Pgas"]*totalgrad*dcd["dy"][:,1]
		dcd["dTdmrat"] = dcd["dy"][:,2]/dcd["dTdm_test"]	# Because doing the reverse will lead to more singularities
		dcd["vcrat"] = dcd["vconv_test"]/dcd["v_conv_st"]
		i += 1


def recalc_wwk04_exploder(dc, lrn=0.95, usevconv=False):
	"""Double-check WWK04 calculations in main code.

	Parameters
	---------
	dc : StarMod data container
		Data container of runaway
	lrn : float
		Fraction of L_nuc enclosed within R_nuc.  Defaults to 0.95
	usevconv : bool
		If true, use dc["stars"][i]["vconv"] rather than "vnuc".  You should
		NOT USE THIS for production-run checking - this is only here because I
		found a bug (now fixed) in the runaway_prod_anal_func.py/find_exploder() 
		function that used vconv rather than vnuc.  Defaults to False, obviously.
	"""

	dc = remove_unfinished_runs(dc)

	wwk04_integral = np.zeros(len(dc["S_c"]))
	for i in range(len(dc["S_c"])):
		i_nuc = min(np.where(dc["stars"][i]["Lnuc"]/max(dc["stars"][i]["Lnuc"]) > lrn)[0])
		dTdr = dc["stars"][i]["dy"][:,2]/dc["stars"][i]["dy"][:,0]		# dT/dr = dT/dm*dm/dr
		if usevconv:
			wwkint = sci_integ.cumtrapz(dTdr + dc["stars"][i]["eps_nuc"]/dc["stars"][i]["cP"]/dc["stars"][i]["vconv"], x=dc["stars"][i]["R"], initial=0.)
		else:
			wwkint = sci_integ.cumtrapz(dTdr + dc["stars"][i]["eps_nuc"]/dc["stars"][i]["cP"]/dc["stars"][i]["vnuc"], x=dc["stars"][i]["R"], initial=0.)
		wwk04_integral[i] = wwkint[i_nuc]

	try:
		return min(np.where(wwk04_integral > 0.)[0])
	except:
		return -1


def recalc_igniter(dc, td=None):

	if not td:
		td = rtc.timescale_data(max_axes=[1e12, 1e12])

	ign_line = td.get_tauneunuc_line()

	for i in range(dc["temp_c"].size):
		if dc["temp_c"][i] > ign_line(dc["dens_c"][i]):
			return i


def recalc_esline():

	td = rtc.timescale_data(max_axes=[1e12, 1e12])
	td.data_p["esgrid"] = 31.7*td.data_p["tau_cdyn"]/td.data_p["tau_nuc"]/td.data_p["delta"]**0.5
	plt.figure()
	cts = plt.contour(td.rho_p, td.T_p, td.data_p["esgrid"], [1.])
	plt.close()

	v = cts.collections[0].get_paths()[0].vertices

	esline = sci_interp.interp1d(v[:,0], v[:,1], kind="linear")

	return [esline, td]


def recalc_es(od, esline=False, td=False):

	if not esline:
		esline = recalc_esline()[0]

	if not td:
		td = rtc.timescale_data(max_axes=[1e12, 1e12])

	rho_od = od["dens_c"].copy(); rho_od[od["dens_c"] < min(td.rho_r)] = min(td.rho_r)
	T_esline = esline(rho_od)
	models_past_esline = np.where(T_esline < od["temp_c"])[0]

	if models_past_esline.size > 0:
		return min(models_past_esline)
	else:
		return -1

##################### TEST SUITE ###########################


def tc_run_suite():
	"""Runs suite of tests to check overall state of code.
	"""

	###### RUN MARTEN CODE COMPARISON ####
	# Set M = 1.15 Msun
	# Coulomb = off
	static_S = [1e7,2e7,3e7,4e7,5e7] + list(10.**np.arange(np.log10(6e7), np.log10(2.2e8*1.01), 0.005))
	od = rw.make_runaway(starmass=1.15*1.9891e33, mymag=False, omega=0., S_arr=static_S, mintemp=1e5, stop_mindenserr=1e-10, mass_tol=1e-6, P_end_ratio=1e-8, simd_userot=False, simd_usegammavar=False, simd_usegrav=False, verbose=True)

	tc_check_vs_marten(od)

	###### CHECK DERIVATIVES ############
