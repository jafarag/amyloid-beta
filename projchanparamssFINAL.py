

import numpy as np
import matplotlib.pyplot as plt
import moose
from collections import namedtuple

EREST_ACT = -70e-3 #: Resting membrane potential

TauInfMinChannelParams = namedtuple('TauInfMinChannelParams', 'T_min T_vdep T_vhalf T_vslope SS_min SS_vdep SS_vhalf SS_vslope')


AlphaBetaChannelParams = namedtuple('AlphaBetaChannelParams','A_rate A_B A_C A_vhalf A_vslope B_rate B_B B_C B_vhalf B_vslope')
Na_m_params = AlphaBetaChannelParams(A_rate = 1e5 * (25e-03 + EREST_ACT),  A_B = -1e05,  A_C = -1.0,  A_vhalf = -25e-03 - EREST_ACT,  A_vslope = -10e-03, B_rate = 4e03, B_B = 0.0, B_C = 0.0, B_vhalf = 0.0 - EREST_ACT, B_vslope = 18e-03)
Na_h_params = AlphaBetaChannelParams(A_rate = 70.0,  A_B = 0.0,  A_C = 0.0,  A_vhalf = 0.0 - EREST_ACT,  A_vslope = 20e-03, B_rate = 1e03, B_B = 0.0, B_C = 1.0, B_vhalf = -30e-03 - EREST_ACT, B_vslope = -10e-03)
K_n_params = AlphaBetaChannelParams(A_rate = 1e4*(10e-03 + EREST_ACT),  A_B = -1e04,  A_C = -1.0,  A_vhalf = -10e-03 - EREST_ACT,  A_vslope = -10e-03, B_rate = 0.125e03, B_B = 0.0, B_C = 0.0, B_vhalf = 0.0 - EREST_ACT, B_vslope = 80e-03)
K_h_params = AlphaBetaChannelParams(A_rate = 70.0,  A_B = 0.0,  A_C = 0.0,  A_vhalf = 0.0 - EREST_ACT,  A_vslope = 20e-03, B_rate = 1e03, B_B = 0.0, B_C = 1.0, B_vhalf = -30e-03 - EREST_ACT, B_vslope = -10e-03)

qfactCaN = 2
CaN_X_params = AlphaBetaChannelParams(A_rate = 304*qfactCaN,
                                      A_B = 0,
                                      A_C = 0.0,
                                      A_vhalf = 0.0,
                                      A_vslope = -14.0e-3,
                                      B_rate = 52800* 14.20003e-3 *qfactCaN,
                                      B_B = 52800*qfactCaN,
                                      B_C = -1.0,
                                      B_vhalf = 14.20003e-3,
                                      B_vslope = 10e-3)

CaN_Y_params = TauInfMinChannelParams(T_min = 70e-3/qfactCaN,
                                      T_vdep = 0,
                                      T_vhalf = 0.0,
                                      T_vslope = -14.0e-3,
                                      SS_min = 0.79,
                                      SS_vdep = 0.21,
                                      SS_vhalf = -74.8e-3,
                                      SS_vslope = 6.5e-3)

qfactKaS = 3
qq = 80000
KaS_X_params = AlphaBetaChannelParams(A_rate = 22057.0306*qfactKaS*qq,
                                      A_B = 0.*qfactKaS*qq,
                                      A_C = 1,
                                      A_vhalf = -0.08789675579999999,
                                      A_vslope = -0.0162951634,
                                      B_rate = 348021313.0*qfactKaS*qq,
                                      B_B = 0.*qfactKaS*qq,
                                      B_C = 1,
                                      B_vhalf =0.39822177799999997,
                                      B_vslope = 0.0218235302)

KaS_Y_params = AlphaBetaChannelParams(A_rate = 25644952.0*qfactKaS*qq*2000,
                                      A_B = 0.*qfactKaS*qq*2000,
                                      A_C = 1.0,
                                      A_vhalf = 1.222,
                                      A_vslope = 0.0645391447,
                                      B_rate = 1.28951669*qfactKaS*qq*2000,
                                      B_B = 0.*qfactKaS*qq*2000,
                                      B_C = 1.0,
                                      B_vhalf = 0.000635602802,
                                      B_vslope = -0.0262013787)
								  
CaL_X_params = AlphaBetaChannelParams(A_rate = -880,
                                        A_B = -220e3,
                                        A_C = -1.0,
                                        A_vhalf = 4.0003e-3,
                                        A_vslope = -7.5e-3,
                                        B_rate = -284,
                                        B_B = 71e3,
                                        B_C = -1.0,
                                        B_vhalf = -4.0003e-3,
                                        B_vslope = 5e-3)

									
qfactCaT = 2
CaT_X_params = AlphaBetaChannelParams(A_rate = 1000*qfactCaT,
                                      A_B = 0.0,
                                      A_C = 0.0,
                                      A_vhalf = 0.0,
                                      A_vslope = -19e-3,
                                      B_rate = 16500 * 81.0003e-3 * qfactCaT,
                                      B_B = 16500*qfactCaT,
                                      B_C = -1.0,
                                      B_vhalf = 81.0003e-3,
                                      B_vslope = 7.12e-3)

#Original inactivation ws too slow compared to activation, made closder the alpha1G
CaT_Y_params = AlphaBetaChannelParams(A_rate = 34000 * 113.0003e-3 * qfactCaT,
                                      A_B = 34000*qfactCaT,
                                      A_C = -1.0,
                                      A_vhalf = 113.0003e-3,
                                      A_vslope = 5.12e-3,
                                      B_rate = 320*qfactCaT,
                                      B_B = 0,
                                      B_C = 0.0,
                                      B_vhalf = 0.0,
                                      B_vslope = -17e-3)
ABParams = namedtuple('ABParams','q10 celsius vhalfl kl qtl a0t vhalft zetat gmt')
H_X_params = ABParams(q10 = 4.5,
                                      celsius = 23,
                                      vhalfl = -81e-3,
									  kl = -8e-3,
									  qtl = 1,
									  a0t = 11,
									  vhalft = -75e-3,
									  zetat = 2.2,
									  gmt = 0.4)

#mho is the same as siemens
Cadepparams = namedtuple('Cadepparams', 'Kd power tau')
SK_Z_params = Cadepparams(Kd = 0.57e-03, power = 5.2, tau = 4.9e-03)
CaCC_Z_params = Cadepparams(Kd = 1.83e-3, power = 2.3, tau = 13e-3)
csettings = namedtuple('csettings', 'Xpow Ypow Zpow Erev name Xparam Yparam Zparam')
caparams = namedtuple('caparams', 'CaBasal CaThick CaTau Bufcapacity caName')
Capar = caparams(CaBasal = 50e-06, CaThick = 1.01e-6, CaTau = 0.010, Bufcapacity = 20, caName = 'Ca')

#Calculating ca rev using nernst potential and concin and conc-out values
#out = 2e-3 in = 50e-9 with units molar
carev = (8.31 * 310 * np.log((2e-3)/(50e-9)))/ (2*96485)
BKChannelParams=namedtuple('BKChannelParams', 'alphabeta K delta')
BK_X_params=[BKChannelParams(alphabeta=480, K=0.18, delta=-0.84),
             BKChannelParams(alphabeta=280, K=0.011, delta=-1.0)]

Na_params = csettings(Xpow = 3, Ypow = 1, Zpow = 0, Erev = 0.045, name ='na', Xparam = Na_m_params,Yparam = Na_h_params, Zparam = [])
K_params = csettings(Xpow = 4, Ypow = 0.0, Zpow = 0, Erev = -0.082, name ='k', Xparam = K_n_params, Yparam = K_h_params, Zparam = [])
BKparam = csettings(Xpow=1, Ypow=0, Zpow=0, Erev=-0.090, name='BKCa', Xparam=BK_X_params,Yparam=[],Zparam=[])
KaS_params = csettings(Xpow=3, Ypow=1, Zpow=0, Erev= -0.090, name='KaS', Xparam = KaS_X_params,Yparam = KaS_Y_params, Zparam = [])
CaLparams = csettings(Xpow=1, Ypow=0, Zpow=0, Erev=carev, name='CaL', Xparam=CaL_X_params,Yparam=[],Zparam=[])
HCNparams = csettings(Xpow=1, Ypow=0, Zpow=0, Erev=-0.030, name='HCN', Xparam= H_X_params,Yparam=[],Zparam=[])
CaCCparam = csettings(Xpow=0, Ypow=0, Zpow=1, Erev= -60e-3, name='CaCC', Xparam = [], Yparam = [], Zparam = CaCC_Z_params)
CaTparams = csettings(Xpow=3, Ypow=1, Zpow=0, Erev=carev, name='CaT', Xparam=CaT_X_params,Yparam=CaT_Y_params,Zparam=[])
CaNparams = csettings(Xpow=2, Ypow=1, Zpow=0, Erev=carev, name='CaN', Xparam=CaN_X_params,Yparam=CaN_Y_params,Zparam=[])
cond_set = {'na': 1200,'k': 360,'CaL':20,'CaN':20,'CaT':20,'HCN':0,'KaS':0,'BKCa':10,'CaCC':50}
#these cond values are 10 times the original values as adapted from the original Gbar equation
chan_set = [Na_params,K_params,CaLparams,CaNparams,CaTparams,HCNparams,KaS_params,BKparam,CaCCparam]
#CaCCparam
#,'SKCa':20,
#,SK_params
#200000

# Shen 2004/Wolf 2005

#: We define the rate parameters, which are functions of Vm as
#: interpolation tables looked up by membrane potential.
#: Minimum x-value for the interpolation table
VMIN = -30e-3 + EREST_ACT
#: Maximum x-value for the interpolation table
VMAX = 120e-3 + EREST_ACT
#: Number of divisions in the interpolation table
VDIVS = 3000







