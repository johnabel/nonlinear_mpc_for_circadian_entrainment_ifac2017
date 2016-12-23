"""
j.h.abel 19/7/2016

adjusting single-step for multi-step optimization
"""

#
#
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os

#3rd party packages
import numpy as np
from scipy import integrate, optimize, stats
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

print 'setup for figure 2: 2hour sampling intervals, 4hour predictive window'

def mpc_problem_fig3(prediction_window):
    """
    function for parallelization. returns the max time at which there is an
    error in phase greater than 0.1rad
    """
    print str(prediction_window)+ ' started'
    shift_value = 12
    shift_val = shift_value*pmodel.T/24.
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
    prcs = pmodel.pPRC_interp
    
    # set up the discretization
    control_dur = 23.7/12 # 2hr
    time_steps = np.arange(0, days*23.7, 2*23.7/24)
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
        print idx, prediction_window, diff**2
        
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
                return np.sum(weights*differences**2)+0.001*np.sum(parameter_shifts**2)
            
            # the sys functions here stop the swarm from printing its results
            xopt, fopt = pso(err, [-0.1]*prediction_window, 
                             [0.0]*prediction_window, 
                            args=(pred_ext_phases, mpc_time_start),
                            maxiter=300, swarmsize=100, minstep=1e-4,
                            minfunc=1e-4)
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
        phases = [pmodel.phase_of_point(pt) for pt in sol[::60]]
        tphases += list(tsol[::60]+time)
        continuous_phases+=list(phases)

    abs_err = np.sqrt(errors)
    ext_phis_output = phase_of_time(shift_time(time_steps))
    simulation_results = [control_inputs, errors, time_steps, mpc_phis, 
                ext_phis_output, mpc_ps, mpc_ts, tphases, continuous_phases]
    print str(shift_value)+ ' completed'
    return simulation_results
    



# parallelize the subfigure generation, use two simulations
inputs = [2,12]
'''
subfig_results = []
with futures.ProcessPoolExecutor(max_workers=2) as executor:
    result = executor.map(mpc_problem_fig3, inputs, chunksize=1)
for idx,res in enumerate(result):
    subfig_results.append(res)

np.save('Data/fig3b15.npy',subfig_results[0])
np.save('Data/fig3c.npy',subfig_results[1])
'''

#load the results
sfr1 = np.load('Data/fig3b.npy')
sfr2 = np.load('Data/fig3c.npy')




# plotting regions =======================

#first plot: part a and b
plo.PlotOptions(ticks='in')



plt.figure(figsize=(3.5,3.0))
gs = gridspec.GridSpec(3,2,  height_ratios=(1.5,1,1))
axes_set1 = [plt.subplot(gs[i,0]) for i in range(3)]
axes_set2 = [plt.subplot(gs[i,1]) for i in range(3)]

def plot_subfig_results(axes, sfr, shift = 0, pred_win=2):
    cx,dx,ex = axes
    control_inputs, errors, time_steps, mpc_phis, ext_phis_output, mpc_ps, mpc_ts, tphases, contphases = sfr
    
    def shift_time(ts, shift_t=1.75*pmodel.T, shift_val=shift):
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
    
    def phase_of_time(times):
        return (times*2*np.pi/pmodel.T)%(2*np.pi)
    
    cx.plot(np.asarray(tphases)/23.7*24.-0.25*24., np.sin(contphases), 'l')
    cx.plot(np.asarray(tphases)/23.7*24.-0.25*24., 
            np.sin(phase_of_time(shift_time(tphases))), 'k--')
    cx.set_xlim([24,96])
    cx.set_xticks([24,48,72,96])
    cx.set_xticklabels([])
    cx.set_ylim([-1.,1.45])
    cx.set_ylabel('sin($\phi$)')
    
    dx.step(control_inputs[1:,0]/23.7*24-.25*24, -control_inputs[:-1,1], '0.7', 
            label='Control Input')
    dx.set_ylim([0,0.11])
    dx.set_xlim([24,96])
    dx.set_xticks([24,48,72,96])
    dx.set_yticks([0,0.05,0.10])
    dx.set_xticklabels([''])
    dx.set_ylabel('u')
    for tl in dx.get_yticklabels():
        tl.set_color('0.7')
    
    dx2 = dx.twinx()
    dx2.plot(np.asarray(tphases)/23.7*24.-0.25*24., 
             -pmodel.pPRC_interp(pmodel._phi_to_t(np.asarray(contphases)))[:,15],'f')    
    dx2.set_xlim([24,96])
    dx2.set_xticks([24,48,72,96])
    plo.format_4pi_axis(dx2, x=False,y=True)
    for tl in dx2.get_yticklabels():
        tl.set_color('f')
    
    ex.step(time_steps[1:-pred_win+1]/23.7*24-.25*24, errors[:], 'k', 
            label='Phase Error')
    ex.set_xlim([24,96])
    ex.set_xticks([24,48,72,96])
    ex.set_yticks([0,5,10])
    ex.set_ylabel('Error (rad$^2$)')
    ex.set_xlabel('Time (h)')
    

plot_subfig_results(axes_set1, sfr1, shift=12)
plot_subfig_results(axes_set2, sfr2, shift=12, pred_win=12)

for ax in axes_set2:
    ax.set_yticklabels([])
    ax.set_ylabel('')

plt.tight_layout(**plo.layout_pad)
plt.show()







