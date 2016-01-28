import sys,os
from numpy import *
sys.path.append("/home/cczhu/GitHubTemp/czerny_wd/")
import StarModSteve as sms
import cPickle as cP
#data = cP.load(open("/media/DataStorage3/Runaway/outputs/od_1pt0Msun_Steve.p", 'r'))
#datai = data["stars"][-1]
#S_old = sms.sprof.entropy_profile(datai["M"], datai["Sgas"], datai["v_conv_st"])
#static_S = [1e7,2e7,3e7,4e7,5e7] + list(10.**arange(log10(6e7), log10(2.2e8*1.01), 0.005))

#densest = 0.8*data["dens_c"][-1]
mystar = sms.mhs_steve(1.9891e33, False, mintemp=1e5, dontintegrate=True, S_old=False)
#mystar.integrate_star(1949726.3159187248, 944033460.06470203, 0., recordstar=True, P_end_ratio=1e-8, ps_eostol=1e-8, outputerr=True)

outerr_code = mystar.getstarmodel(densest=1949726.3159187248, S_want=179122957.13507387, P_end_ratio=1e-8, ps_eostol=1e-8)

#mystar.integrate_star(4652844.8120659525, 944033460.06470203, 0., recordstar=True, P_end_ratio=1e-7, ps_eostol=1e-7, outputerr=True)
#mystar.getstarmodel(densest=densest, S_want=static_S[90], P_end_ratio=1e-7, ps_eostol=1e-7, deltastepcoeff=0.1)
#mystar.integrate_star(4652844.8120659525, 944033460.06470203, 0., recordstar=True, P_end_ratio=1e-7, ps_eostol=1e-7, outputerr=True)

#make_runaway_steve(starmass=1.2*1.9891e33, mymag=False, omega=0., omega_run_rat=0.8, S_arr=10**np.arange(7.5,8.2,0.25), mintemp=1e5, S_old=False, mass_tol=1e-6, P_end_ratio=1e-8, densest=False, L_tol=1e-2, keepstars=False, omega_crit_tol=1e-3, omega_warn=10., verbose=True)
