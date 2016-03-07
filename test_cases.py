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
import StarMod as Star
import runaway as rw

##################### SUPPORT FUNCTIONS ####################

def tc_get_marten_WD(dens_c, temp_c):
	""" Obtain WD profile from Marten's HelmholtzWD code.
	"""
	os.system("./martenwdmaker/helmholtzwd {0:.6e} {1:.6e} > martentemp.txt".format(dens_c, temp_c))

	f = open("martentemp.txt", 'r')
	data2 = np.loadtxt(f)

	out = {}
	out["R"] = data2[:,0]
	out["dens"] = data2[:,1]
	out["M"] = data2[:,2]
	
	f.close()
	os.system("rm martentemp.txt")

	return out


def tc_uneven_arrays_ratio(X1, Y1, X2, Y2):
	"""Checks the ratio of two profiles ([X1, Y1] and [X2, Y2]) that have different lengths.

	Arguments:
	X1/Y1: x and y values of profile 1
	X2/Y2: x and y values of profile 2
	"""
	if max(X1) > max(X2):
		X_spline = X1; Y_spline = Y1
		X_use = X2; Y_use = Y2
	else:
		X_spline = X2; Y_spline = Y2
		X_use = X1; Y_use = Y1
	Ysout = sci_interp.UnivariateSpline(X_spline, Y_spline)
	Y_num = Ysout(X_use)
	return [X_use, Y_num/Y_use]


def tc_checkmodeltol(od):
	"""Checks if mass and angular momentum are self-consistent with mass tolerance.  Magnetic flux conservation should be checked too, of course, but that is not formally bounded in the runaway code, so should be checked separately.
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

##################### TEST SUITE ###########################

def tc_check_vs_marten(od, tol=1e-3):
	"""Checks data instance against Marten's code.
	"""

	for i in range(len(od["stars"])):
		marten = tc_get_marten_WD(od["stars"]["rho"][0], od["stars"]["T"][0])
		rho_dev = abs(od["stars"]["rho"] - marten["dens"])/od["stars"]["rho"]
		if max(rho_dev) > tol:
			raise AssertionError("ERROR: maximum (rho - rho_marten)/rho = {0:.3e} > {1:.3e}".format(max(rho_dev), tol))


def tc_run_suite():
	"""Runs suite of tests to check overall state of code.
	"""

	###### RUN MARTEN CODE COMPARISON ####
	# Set M = 1.15 Msun
	# Coulomb = off

	static_S = [1e7,2e7,3e7,4e7,5e7] + list(10.**arange(np.log10(6e7), np.log10(2.2e8*1.01), 0.005))
	od = rw.make_runaway(starmass=1.15*1.9891e33, mymag=False, omega=0., S_arr=static_S, mintemp=1e5, stop_mindenserr=1e-10, derivtype="sim", mass_tol=1e-6, P_end_ratio=1e-8, simd_userot=False, simd_usegammavar=False, simd_usegrav=False, simd_suppress=False, verbose=True)

	tc_check_vs_marten(od)
