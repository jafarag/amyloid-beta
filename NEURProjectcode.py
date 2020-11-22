

import numpy as np
import matplotlib.pyplot as plt
import moose
from collections import namedtuple
import projchanparams as cp


#: Minimum x-value for the interpolation table
VMIN = -100e-3 
#: Maximum x-value for the interpolation table
VMAX = 500e-3 
#: Number of divisions in the interpolation table
VDIVS = 3000



#if type(chanparams) == 'AlphaBetaChannelParams':
#if isinstance(params,AlphaBetaChannelParams):
def sigmoid(x,xmin,xmax,xvhalf,xslope):
    return xmin+xmax/(1+np.exp((x-xvhalf)/xslope))



def quadratic(x,xmin,xmax,xvhalf,xslope):
    tau1 = xmax/(1+np.exp((x-xvhalf)/xslope))
    tau2 = 1/(1+np.exp((x-xvhalf)/-xslope))
    tau_x = xmin+tau1*tau2
    return tau_x


def make_sigmoid_gate(params,Gate,VDIVS = 3000, VMIN = -100e-3 , VMAX = 50e-3):
    Gate.min = VMIN
    Gate.max = VMAX
    Gate.divs = VDIVS
    v = np.linspace(VMIN, VMAX, VDIVS)
    if params.T_power==2:
        print('making quadratic gate', Gate.path)
        tau = quadratic(v,params.T_min,params.T_vdep,params.T_vhalf,params.T_vslope)
    else:
        tau = sigmoid(v,params.T_min,params.T_vdep,params.T_vhalf,params.T_vslope)
    minf = sigmoid(v,params.SS_min,params.SS_vdep,params.SS_vhalf,params.SS_vslope)
    Gate.tableA = minf/tau
    Gate.tableB = 1/tau


def interpolate_values_in_table(tabA, V_0, l=40):
    la = len(tabA)
    V = np.linspace(VMIN,VMAX,la)
    idx =  abs(V-V_0).argmin()
    min_idx=max(idx-l,0)
    max_idx=min(idx+l,len(tabA)-1)
    #print('in interp, len of tabA',len(tabA),'V0',V_0,'idx',idx,'+l',idx+l,'min',min_idx,'max',max_idx)
    A_min = tabA[min_idx]
    V_min = V[min_idx]
    A_max = tabA[max_idx]
    V_max = V[max_idx]
    a = (A_max-A_min)/(V_max-V_min)
    b = A_max - a*V_max
    tabA[min_idx:max_idx] = V[min_idx:max_idx]*a+b
    return tabA


def calc_V0(rate,B,C,vhalf,vslope):
        newrate = rate
        delta = rate - B * vhalf
        if delta > 1e-10 or delta < -1e-10:
            newrate= B * vhalf
        V_0 = vslope*np.log(-C)-vhalf
        return newrate,V_0


def fix_singularities(Params, Gate):
    #This needs to be extended to work with standardMooseTauInfparams
    if Params.A_C < 0:
        #print('fix_sing for',Params,'len of table',len(Gate.tableA))
        A_rate,V_0=calc_V0(Params.A_rate,Params.A_B,Params.A_C,Params.A_vhalf,Params.A_vslope)
        if -100e-3 < V_0 < 50e-03:
            #change values in tableA and tableB, because tableB contains sum of alpha and beta
            Gate.tableA = interpolate_values_in_table(Gate.tableA, V_0)
            Gate.tableB = interpolate_values_in_table(Gate.tableB, V_0)
	return A_rate, V_0		

	
def BKchan_proto(chanparams, VDIVS = 3000, VMIN = -100e-3 , VMAX = 50e-3, CAMIN = 0, CAMAX = 1, CADIVS = 5000):
	ZFbyRT=2*96520/(8.3134*(23+273.15))
	v_array = np.linspace(VMIN, VMAX, VDIVS)
	ca_array = np.linspace(CAMIN, CAMAX, CADIVS)
	#set up the two dimensional gating matrix:
	gatingMatrix = []
	for i,pars in enumerate(chanparams.Xparam):
		Vdepgating=pars.K*np.exp(pars.delta*ZFbyRT*v_array)
		#These assignments are specific to the BK channel, calculate 2D array of gating values
		if i == 0:
	#This is the forward rate constant for a two state channel
			gatingMatrix.append(pars.alphabeta*ca_array[None,:]/(ca_array[None,:]+pars.K*Vdepgating[:,None]))
		else:
	#this is backward rate constant for a two state channel
			gatingMatrix.append(pars.alphabeta/(1+ca_array[None,:]/pars.K*Vdepgating[:,None]))
	#adding forward rate to backward rate  gives “alpha+beta” which = 1/tau
		gatingMatrix[i] += gatingMatrix[0]
	
	chan = moose.HHChannel2D('/library/'+chanparams.name) # two dimensional tabulated channel gating
	chan.Xpower = chanparams.Xpow
	chan.Ek=chanparams.Erev
	chan.Xindex="VOLT_C1_INDEX"      # critical for correctly using voltage and calcium
	xGate = moose.HHGate2D(chan.path + '/gateX')
	xGate.xminA=xGate.xminB=VMIN
	xGate.xmaxA=xGate.xmaxB=VMAX
	xGate.xdivsA=xGate.xdivsB=VDIVS
	xGate.yminA=xGate.yminB=CAMIN
	xGate.ymaxA=xGate.ymaxB=CAMAX
	xGate.ydivsA=xGate.ydivsB=CADIVS
	xGate.tableA=gatingMatrix[0]	# assign gatingMatrix to tables, [0] is forw rate = ss/tau
	xGate.tableB=gatingMatrix[1]	#[1] is 1/tau
	return chan

def chan_proto(chanparams, VDIVS = 3000, VMIN = -100e-3 , VMAX = 50e-3, CAMIN = 0, CAMAX = 1, CADIVS = 5000):
	lib = moose.Neutral('/library') 
	chan = moose.HHChannel('/library/'+chanparams.name)  # create the channel
	chan.Ek = chanparams.Erev
	chan.Xpower = chanparams.Xpow
	if chan.Xpower > 0:
		xGate = moose.HHGate(chan.path + '/gateX') # create the activation gating variable
		if chanparams.name ==  'CaL':
			xGate.setupAlpha(chanparams.Xparam + (VDIVS, VMIN, VMAX))
			#fix_singularities(chanparams.Xparam, xGate)
		elif chanparams.name ==  'HCN':
			xGate.min = VMIN
			xGate.max = VMAX
			varray = np.linspace(VMIN, VMAX, VDIVS)
			qt = chanparams.Xparam.q10**((chanparams.Xparam.celsius-33)/10)
			inf_x = 1/(1 + np.exp(-(varray-chanparams.Xparam.vhalfl)/chanparams.Xparam.kl))
			a = np.exp(0.0378*chanparams.Xparam.zetat*(varray-chanparams.Xparam.vhalft))
			bett = np.exp(0.0378*chanparams.Xparam.zetat*chanparams.Xparam.gmt*(varray-chanparams.Xparam.vhalft))
			tau_x = bett/(chanparams.Xparam.qtl*qt*chanparams.Xparam.a0t*(1+a)) 
			xGate.tableA=inf_x / tau_x
			xGate.tableB=1 / tau_x
			#setupAlpha automatically creates tables of a and b between VMIN and VMAX
		#elif chanparams.name == 'na':	
		#	make_sigmoid_gate(chanparams.Xparam,xGate,VDIVS = 3000, VMIN = -100e-3 , VMAX = 50e-3)
		else:
			xGate.setupAlpha(chanparams.Xparam + (VDIVS, VMIN, VMAX))
	chan.Ypower = chanparams.Ypow
	if chan.Ypower > 0:    #optional inactivation gate
		yGate = moose.HHGate(chan.path + '/gateY')
		#if chanparams.name == 'na':
		#	make_sigmoid_gate(chanparams.Yparam,yGate,VDIVS = 3000, VMIN = -100e-3 , VMAX = 50e-3)
		if chanparams.name == 'CaN':
			yGate.min = VMIN
			yGate.max = VMAX
			varray = np.linspace(VMIN, VMAX, VDIVS)
			tau = sigmoid(varray,chanparams.Yparam.T_min,chanparams.Yparam.T_vdep,chanparams.Yparam.T_vhalf,chanparams.Yparam.T_vslope)
			minf = sigmoid(varray,chanparams.Yparam.SS_min,chanparams.Yparam.SS_vdep,chanparams.Yparam.SS_vhalf,chanparams.Yparam.SS_vslope)
			yGate.tableA=minf / tau
			yGate.tableB=1 / tau
		else:
			yGate.setupAlpha(list(chanparams.Yparam) + [VDIVS, VMIN, VMAX])
	if chanparams.Zpow > 0:
		chan.Zpower = chanparams.Zpow   #normal channel stuff
		zgate = moose.HHGate(chan.path + '/gateZ')
		ca_array = np.linspace(CAMIN, CAMAX, CADIVS)
		zgate.min=CAMIN         #specific to calcium dependent channels
		zgate.max=CAMAX
		caterm=(ca_array/chanparams.Zparam.Kd)**chanparams.Zparam.power
		inf_z=caterm/(1+caterm)
		tau_z=chanparams.Zparam.tau*np.ones(len(ca_array))
		zgate.tableA=inf_z / tau_z
		zgate.tableB=1 / tau_z
		chan.useConcentration=True
	return chan



def chanlib(chan_set):
    if not moose.exists('/library'):
        #only create ‘/library’ if it has not been created already
        lib = moose.Neutral('/library')
    for params in chan_set:
 	    chan=chan_proto(params)

  
def chanlib2(chan_set):
    if not moose.exists('/library'):
        #only create ‘/library’ if it has not been created already
        lib = moose.Neutral('/library')
    for params in chan_set:
        if params.name == 'BKCa':
            chan=BKchan_proto(params)
        else:
            chan=chan_proto(params)



#create comp here


def create_comp(celln,name,Len,dia,RM,CM,RA):
	comp = moose.Compartment(celln.path+'/'+str(name))
	comp.diameter = dia
	comp.length = Len
	x_area = np.pi*dia*dia/4.0
	s_area  = np.pi*dia*Len
	comp.Rm = 1/(RM*s_area)
	comp.Cm = CM*s_area
	comp.Ra = RA*Len/x_area
	comp.Em = -0.07 + 10.613e-3
	comp.initVm = -0.07
	return comp



Dist_dendd = {'KaS': {'factor': 0.3, 'slope': 100}, 'HCN': {'factor': 0.0005, 'slope': 3}}
def channel_to_compartment(cond_set,comp,Dist_dend):
	distance=np.sqrt(comp.x*comp.x+comp.y*comp.y+comp.z*comp.z)
	for channame, cond in cond_set.items():
		SA=np.pi*comp.length*comp.diameter
		proto = moose.element('/library/'+channame)
		chan = moose.copy(proto, comp, channame)[0]
		if channame in Dist_dend.keys():
			if channame == 'KaS':
				if comp.diameter < 0.8e-6:
					cond = 2000*Dist_dend[str(channame)]['factor']*((1+3*distance/Dist_dend[str(channame)]['slope']))
				else:
					cond = 2000*Dist_dend[str(channame)]['factor']*((1+1*distance/Dist_dend[str(channame)]['slope']))
				print(comp.name, cond)
			elif channame == 'HCN':
				cond = Dist_dend[str(channame)]['factor']*(1+Dist_dend[str(channame)]['slope']/distance)
		else:
			X = []	
		chan.Gbar = cond * SA
		m=moose.connect(chan, 'channel', comp, 'channel')
    
	

	
#channel_to_compartment(cond_set,comp)


def CaProto(caparams):
    if not moose.exists('/library'):
        lib=moose.Neutral('/library')
    poolproto=moose.CaConc('/library'+ caparams.caName)
    poolproto.CaBasal=caparams.CaBasal
    poolproto.ceiling=1
    poolproto.floor=0
    poolproto.thick=caparams.CaThick
    poolproto.tau=caparams.CaTau
    return poolproto


def add_calcium(cellname,caparams):
    caproto=CaProto(caparams)
    for comp in moose.wildcardFind('%s/#[TYPE=Compartment]'%(cellname)):
        capool= moose.copy(caproto, comp, caparams.caName)
        print capool.path
        capool.length = comp.length
        capool.diameter= comp.diameter
        radius = capool.diameter/2
        if radius<=2*caproto.thick:
            capool.thick = radius/4
        vol=np.pi*(radius**2 - (radius-caproto.thick)**2)*capool.length
        capool.B=1/(96485*vol*2)/caparams.Bufcapacity
    return 


def add_calcium2(comp,caparams):
    caproto=CaProto(caparams)
    capool= moose.copy(caproto, comp, caparams.caName)
    print capool.path
    capool.length = comp.length
    capool.diameter= comp.diameter
    radius = capool.diameter/2
    if radius<=2*caproto.thick:
        capool.thick = radius/4
    vol=np.pi*(radius**2 - (radius-caproto.thick)**2)*capool.length
    capool.B=1/(96485*vol*2)/caparams.Bufcapacity
    return 



def connect_cal2chan(chan_names,chan_types,cellname,calname,synparams):
	for comp in moose.wildcardFind('%s/#[TYPE=Compartment]'%(cellname)):
		capool=moose.element(comp.path+'/'+calname)
		for cn,ct in zip(chan_names,chan_types):
			chan=moose.element(comp.path+'/'+cn)
			if  ct=='ca_perm':
				m = moose.connect(chan, 'IkOut', capool, 'current')
			elif ct=='ca_dep':
				m = moose.connect(capool,'concOut', chan, 'concen')
        for key in synparams.items():
            if key=='nmda':
               nmdachan=moose.element(comp.path+'/'+key)
               m = moose.connect(nmdachan, 'ICaOut', capool, 'current')
               m = moose.connect(capool,'CaConc', nmdachan, 'setIntCa')
	return

def connect_cal2chan2(chan_names,chan_types,comp,calname,synparams):
	capool=moose.element(comp.path+'/'+calname)
	for cn,ct in zip(chan_names,chan_types):
		chan=moose.element(comp.path+'/'+cn)
		if  ct=='ca_perm':
			m = moose.connect(chan, 'IkOut', capool, 'current')
		elif ct=='ca_dep':
			m = moose.connect(capool,'concOut', chan, 'concen')
        for key in synparams.items():
            if key=='nmda':
               nmdachan=moose.element(comp.path+'/'+key)
               m = moose.connect(nmdachan, 'ICaOut', capool, 'current')
               m = moose.connect(capool,'CaConc', nmdachan, 'setIntCa')
	return


Cond_set={'Na': {(0,30e-6): 115, (30e-6,1): 11}, 'KDr': {(0,30e-6): 36,(30e-6,1):3.6}}
Dist_dendd = {'KaS': {'factor': 0.3, 'slope': 100}, 'HCN': {'factor': 0.0005, 'slope': 3}}
def update_cell(ChanDict,CondSet,name,pfile, caparams, chan_names,chan_types,Dist_dend, synparams):
	chanlib2(ChanDict)
	cell= moose.loadModel(pfile,name)
	for comp in moose.wildcardFind('%s/#[TYPE=Compartment]'%(name)):
		distance=np.sqrt(comp.x*comp.x+comp.y*comp.y+comp.z*comp.z)
		SA=np.pi*comp.length*comp.diameter
		channel_to_compartment(CondSet,comp,Dist_dend)
		add_calcium2(comp,caparams)
		connect_cal2chan2(chan_names,chan_types,comp,caparams.caName,synparams)
		if comp.name != '1_1': 
			if distance < 350e-6:
				comp.Rm =  0.0316227766017/(SA)
			else:
				comp.Rm =  0.316227766017/(SA)
		else:
			print('soma')
	print('cell updated!')
	return cell




def create_pulse(comp):
 	pulse = moose.PulseGen('/model/stimulus')
	pulse.delay[0] = 0.01
	pulse.level[0] = 7.2e-9
	pulse.width[0] = 200e-03
	pulse.delay[1] = 1e9
	moose.connect(pulse, 'output', comp, 'injectMsg')
	data = moose.Neutral('/data')
	pulse_tab = moose.Table('/data/current')
	moose.connect(pulse_tab, 'requestOut', pulse, 'getOutputValue')
	return pulse_tab


def create_output(comp):
	vmtab = moose.Table('/data/Vm'+comp.name)
	moose.connect(vmtab, 'requestOut', comp, 'getVm')
	return vmtab
 

 
def ca_output(comp):
	catab = moose.Table('/data/Ca'+comp.name)
	compca = moose.element(comp.path+'/Ca')
	moose.connect(catab, 'requestOut', compca, 'getCa')
	return catab


def adjust_clocks(simdt,plotdt):
	for i in range(10):
		moose.setClock(i,simdt)
	moose.setClock(8,plotdt)

Synparams = {}
mgparams={'A':(1/6.0), 'B':(1/80.0), 'conc': 1.4}
Synparams['ampa']={'Erev': 5e-3,'tau1': 1.0e-3,'tau2': 5e-3,'Gbar': 2e-9}
Synparams['nmda']={'Erev': 5e-3, 'tau1': 1.1e-3, 'tau2': 37.5e-3, 'Gbar': 2e-9, 'mgparams': mgparams}
#Synparams['gaba']={'Erev': -e-3,'tau1': 1.0e-3,'tau2': 5e-3,'Gbar': 2e-9}
 

#in the future:
#dict[a] = b -> dict{'a':b}
#use print(....) to confirm if compartments/channels created. saves time rather than usign show msg
	
if __name__ == '__main__':
   model = moose.Neutral('/model')
   cll = moose.Neutral('/neuron')	
   chan_namess=['CaL', 'CaN', 'CaT','BKCa','CaCC']		#'SKCa', 
   chan_typess=['ca_perm', 'ca_perm', 'ca_perm','ca_dep','ca_dep']	#'ca_dep', 
   chan_set = [cp.Na_params,cp.K_params,cp.BKparam,cp.CaLparams]
   nseg = 1			#change nseg to 1 for soma only cell
   pfile4 = 'somadendonly.p'
   pfile5 = 'cell_shift.p'
   cellcont = 'cell1'
   cell1 = update_cell(cp.chan_set,cp.cond_set,cellcont,pfile5, cp.Capar,chan_namess,chan_typess,Dist_dendd, Synparams)
   soma = moose.element('cell1/1_1')
   obdend = moose.element('cell1/1471_4')
   proxdend = moose.element('cell1/1469_4')
   create_pulse(soma) 
   simtime = 0.05
   simdt = 0.25e-5
   plotdt = 0.25e-3
   data = moose.Neutral('/data')
   vm3_tab = create_output(proxdend)
   vm2_tab = create_output(soma)
   vm1_tab = create_output(obdend)
   ca3_tab = ca_output(proxdend)
   ca2_tab = ca_output(soma)
   ca1_tab = ca_output(obdend)
   adjust_clocks(simdt,plotdt)
   hsolve = moose.HSolve(cell1.name + '/hsolve')
   hsolve.dt=simdt
   # Compartment is transformed into zombiecompartment after below statement.
   hsolve.target = soma.name
   #log.info("Using HSOLVE for {} clock {}", hsolve.path, hsolve.tick)
   moose.reinit()
   moose.start(simtime)
   ts = np.linspace(0, simtime, len(vm2_tab.vector))
   plt.plot(ts, ca1_tab.vector *1e-3, label = 'cell 1 oblique dend (mV)')
   plt.plot(ts, ca2_tab.vector *1e-3, label='cell 1 soma Vm (mV)')
   plt.plot(ts, ca3_tab.vector *1e-3, label = 'cell 1 proximal dend (mV)')
   plt.ylabel('Calcium (mM)')
   plt.xlabel('Time (s)')
   plt.legend()
   plt.title('Control')
   plt.ion()
   plt.show()
