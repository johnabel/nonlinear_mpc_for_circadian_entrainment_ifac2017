"""
j.h.abel 19/7/2016

adjusting single-step for multi-step optimization
"""

#
#
# -*- coding: utf-8 -*-
#basic packages
from __future__ import division
import sys
import os

#3rd party packages
import numpy as np
from scipy import integrate, optimize, stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import casadi as cs
from pyswarm import pso
from concurrent import futures

#local imports
from LocalModels.hirota2012 import model, param, y0in
from LocalImports import LimitCycle as lco
from LocalImports import PlotOptions as plo
from LocalImports import Utilities as uts


pmodel = lco.Oscillator(model(), param, y0in)
pmodel.calc_y0()
pmodel.corestationary()
pmodel.limit_cycle()
pmodel.find_prc()

sim=False # flag for whether or not to simulate model
def mpc_problem_time(inputs):
    """
    function for parallelization. returns the max time at which there is an
    error in phase greater than 0.1rad
    """
    trial_index, sampling_len, shift_val = inputs
    shift_val = shift_val*pmodel.T/24.
    # set up times, and phases
    days = 7
    shift_day = 1.75
    times = np.linspace(0,days*23.7, days*1200+1)
    def phase_of_time(times):
        return (times*2*np.pi/pmodel.T)%(2*np.pi)
        
    def shift_time(ts, shift_t=shift_day*pmodel.T, shift_val=shift_val):
        if np.size(ts) is 1:
             if ts >= shift_t:
                 return ts+shift_val
             else:
                 return ts
        elif np.size(ts) > 1:
            ts = np.copy(ts)
            ts[np.where(ts>=shift_t)]+=shift_val
            return np.asarray(ts)
        else:
            print "unknown type supplied"
    
    
    
    
    # create the ref and uncontrolled traj
    unpert_p = pmodel.lc(times)[:,0]
    pert_p = pmodel.lc(pmodel._phi_to_t(phase_of_time(shift_time(times))))[:,0]
    
    
    # control inputs
    control_vectors = [pmodel.pdict['vdCn']]
    prcs = pmodel.pPRC_interp#,                   pmodel.sPRC_interp(pmodel.pdict['vaC1P'])]
    
    # set up the discretization
    control_dur = 23.7/24*24 # 24hr
    prediction_window = int(control_dur/(sampling_len*23.7/24))
    time_steps = np.arange(0, days*23.7, sampling_len*23.7/24)
    tstep_len = time_steps[1]
    control_inputs = np.zeros([len(time_steps)-prediction_window,
                               len(control_vectors)+1])
    single_step_ts = np.linspace(0, tstep_len, 100)
    
    
    
    # initialize the problem
    segment_ystart = y0in
    mpc_ts = []
    mpc_ps = []
    mpc_phis = []
    mpc_pts  = []
    pred_phis = [0.]
    errors = []
    continuous_phases = []
    tphases = []
    
    for idx,time in enumerate(time_steps[:-prediction_window]):
        #calculate current phase of oscillator
        mpc_phi=None
        while mpc_phi is None:
            # this sometimes fails to converge and I do not know why
            try:  mpc_phi = pmodel.phase_of_point(segment_ystart)
            except: 
                mpc_phi = 0
                print segment_ystart
                
                
        mpc_phis.append(mpc_phi)
        mpc_pts.append(segment_ystart)
        mpc_time_start = pmodel._phi_to_t(mpc_phi) # used for calculating steps
        control_inputs[idx,0] = time
        
        # get the absolute times, external times, external phases, and predicted 
        # phases. predicted phases are what's used for the mpc
        abs_times = time_steps[idx:idx+prediction_window+1]
        abs_phases = phase_of_time(abs_times)
        ext_times = shift_time(abs_times)
        ext_phases = phase_of_time(ext_times)
        # the next predicted phases do not anticipate the shift
        pred_ext_times = ext_times[0] + time_steps[1:prediction_window+1]
        pred_ext_phases = pmodel._t_to_phi(pred_ext_times)%(2*np.pi)
        
        # calculate the error in phase at the current step
        p1=mpc_phi; p2=ext_phases[0]
        diff = np.min([(p1-p2)%(2*np.pi), (p2-p1)%(2*np.pi)])
                
        errors.append(diff**2)
        print idx, diff**2
        
        # only calculate if the error matters
        if diff**2 > 0.01:
            # define the objective fcn
            def err(parameter_shifts, future_ext_phases, mpc_time):
                # assert that args are good
                assert len(parameter_shifts)==len(future_ext_phases), \
                            "length mismatch between u, phi_ext"
                osc_phases = np.zeros(len(parameter_shifts)+1)
                osc_phases[0] = mpc_phi
                
                for i,pshift in enumerate(parameter_shifts):
                    # next oscilltor phase = curr phase + norm prog +integr pPRC
                    def dphidt(phi, t0):
                        return 2*np.pi/(pmodel.T)+pshift*prcs(pmodel._phi_to_t(phi))[:,control_vectors[0]]
                    osc_phases[i+1] = integrate.odeint(dphidt, osc_phases[i], single_step_ts)[-1]
                    
                #calc difference
                p1=osc_phases[1:]; p2=future_ext_phases 
                differences = np.asarray([(p1-p2)%(2*np.pi), (p2-p1)%(2*np.pi)]).min(0)
                differences = stats.threshold(differences,threshmin=0.1)
                #quadratic cost in time
                weights = (np.arange(len(differences))+1)
                return np.sum(weights*(differences)**2)+np.sum(weights*parameter_shifts**2)
            
            # the sys functions here stop the swarm from printing its results
            sys.stdout = open(os.devnull, "w") # part 1
            xopt, fopt = pso(err, [-0.1]*prediction_window, 
                             [0.0]*prediction_window, 
                            args=(pred_ext_phases, mpc_time_start),
                            maxiter=100, swarmsize=500, minstep=1e-4,
                            minfunc=1e-4)
            sys.stdout = sys.__stdout__ # part 2
            opt_shifts = xopt 
            opt_first_step = opt_shifts[0]
    
        else:
            # no action necessary
            opt_shifts = np.zeros(prediction_window)
            opt_first_step = opt_shifts[0]
        
        # simulate everything forward for one step
        control_inputs[idx,1] = opt_first_step
        mpc_opt_param = np.copy(param)
        mpc_opt_param[pmodel.pdict['vdCn']]+=opt_first_step
        mpc_model = lco.Oscillator(model(), mpc_opt_param, segment_ystart)
        sol = mpc_model.int_odes(tstep_len, numsteps=500)
        tsol = mpc_model.ts
        
        # add the step results to the overall results
        segment_ystart = sol[-1]
        mpc_ts+=list(tsol+time)
        mpc_ps+=list(sol[:,0])

    abs_err = np.sqrt(errors)
    print trial_index
    try:
        times_where_syncd = time_steps[np.where(abs_err<0.1)]
        time_to_sync = np.min(times_where_syncd[np.where(
                    times_where_syncd>(1.75*23.7-0.0001))]) - 1.75*23.7
        np.round(time_to_sync,3)
    except:
        # if it's outside that range we don't want to fail
        time_to_sync = -2
    return sampling_len, shift_val*24./pmodel.T, time_to_sync*24./pmodel.T


# generate inputs for the resync time function
inputs = []
idx = 0
for sampling_len in [1,2,3,4,6,8,12]:
    for shift_val in range(-12,13,2):
        inputs.append([idx, sampling_len, shift_val])
        idx+=1

# only simulate if commanded
if sim==True:
    timer = uts.laptimer()
    num_cpus = 6
    with futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
        result = executor.map(mpc_problem_time, inputs, chunksize=1)
    
    results1 = []
    errors1 = []
    for idx,res in enumerate(result):
        try:
            results1.append(res)
        except:
            errors1.append(inputs[idx])
        
    results1 = np.asarray(results1)
    np.save('Data/sampling_timetosync.npy',
            results1)

    print timer()
    
    x=np.unique(results[:,1]) # window
    y=np.unique(results[:,0]) # delphi
    V=values.reshape(len(y),len(x))
else:
    V = np.load('Data/sampling_timetosync_values.npy')




plo.PlotOptions(ticks='in')
plt.figure()

ax = plt.subplot()

ys = np.arange(8)
xs = np.hstack([x,[14]])
cbar = ax.pcolormesh(xs-1,ys,V,cmap='CMRmap',alpha=1,vmin=0, vmax=100)
ax.set_xlabel(r'$\Delta\phi$ (h)')
ax.set_ylabel(r'Samples-per-24h Window')
ax.set_xlim([-13, 13])
ax.set_yticks(np.arange(1,9)-0.5)
ax.set_yticklabels([24,12,8,6,4,3,2,1])

ax.set_xticks(np.arange(-12,13,2))

# mark region with no resync
ax.add_patch(mpl.patches.Rectangle((-13, 7),
                        26, 2, hatch='//', color='k', fill=False, snap=False))
ax.text(0,7.4,'Does Not Re-entrain',horizontalalignment='center')


plt.colorbar(cbar, ticks = [0,25,50,75,100], 
             label='Time to Re-entrain (h)')

plt.tight_layout(**plo.layout_pad)
plt.show()










