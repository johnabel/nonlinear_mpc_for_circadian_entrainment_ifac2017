# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:53:42 2014
 
@author: John H. Abel

This file will be my repository of classes and functions to call when
solving models. Any general methods will be here.

"""

from __future__ import division
import cPickle as pickle
import numpy as np
import casadi as cs
import pylab as pl
import matplotlib.pyplot as plt
import Utilities as jha
import pdb
from scipy import signal
from scipy.interpolate import splrep, splev, UnivariateSpline


class Oscillator(object):
    """
    This circadian oscillator class is for deterministic ODE simulations.
    """
    
    def __init__(self, model, param, y0=None, period_guess=24.):
        """
        Setup the required information.
        ----
        model : casadi.sxfunction
            model equations, sepecified through an integrator-ready
            casadi sx function
        paramset : iterable
            parameters for the model provided. Must be the correct length.
        y0 : optional iterable
            Initial conditions, specifying where d(y[0])/dt = 0
            (maximum) for the first state variable.
        """
        self.model = model
        self.modifiedModel()
        self.neq = self.model.input(cs.DAE_X).size()
        self.np = self.model.input(cs.DAE_P).size()
        
        self.model.init()
        self.param = param

        self.jacp = self.model.jacobian(cs.DAE_P,0); self.jacp.init()
        self.jacy = self.model.jacobian(cs.DAE_X,0); self.jacy.init()
        
        self.ylabels = [self.model.inputExpr(cs.DAE_X)[i].getName()
                        for i in xrange(self.neq)]
        self.plabels = [self.model.inputExpr(cs.DAE_P)[i].getName()
                        for i in xrange(self.np)]
        
        self.pdict = {}
        self.ydict = {}
        
        for par,ind in zip(self.plabels,range(0,self.np)):
            self.pdict[par] = ind
            
        for par,ind in zip(self.ylabels,range(0,self.neq)):
            self.ydict[par] = ind
        
        self.inverse_ydict = {v: k for k, v in self.ydict.items()}
        self.inverse_pdict = {v: k for k, v in self.pdict.items()}
        
        self.intoptions = {
            'y0tol'            : 1E-3,
            'bvp_ftol'         : 1E-10,
            'bvp_abstol'       : 1E-12,
            'bvp_reltol'       : 1E-10,
            'sensabstol'       : 1E-11,
            'sensreltol'       : 1E-9,
            'sensmaxnumsteps'  : 80000,
            'sensmethod'       : 'staggered',
            'transabstol'      : 1E-6,
            'transreltol'      : 1E-6,
            'transmaxnumsteps' : 5000,
            'lc_abstol'        : 1E-11,
            'lc_reltol'        : 1E-9,
            'lc_maxnumsteps'   : 40000,
            'lc_res'           : 200,
            'int_abstol'       : 1E-10,
            'int_reltol'       : 1E-8,
            'int_maxstepcount' : 40000,
            'constraints'      : 'positive'
                }

        if y0 is None:
            self.y0 = 5*np.ones(self.neq)
            self.calc_y0(25*period_guess)
        else: self.y0 = np.asarray_chkfinite(y0)     
    
    # shortcuts
    def _phi_to_t(self, phi): return phi*self.T/(2*np.pi)
    def _t_to_phi(self, t): return (2*np.pi)*t/self.T
    
    
    def modifiedModel(self):
        """
        Creates a new casadi model with period as a parameter, such that
        the model has an oscillatory period of 1. Necessary for the
        exact determinination of the period and initial conditions
        through the BVP method. (see Wilkins et. al. 2009 SIAM J of Sci
        Comp)
        """

        pSX = self.model.inputExpr(cs.DAE_P)
        T = cs.SX.sym("T")
        pTSX = cs.vertcat([pSX, T])
        
        t = self.model.inputExpr(cs.DAE_T)
        sys = self.model.inputExpr(cs.DAE_X)
        ode = self.model.outputExpr()[0]*T
        
        self.modlT = cs.SXFunction(
            cs.daeIn(t=t,x=sys,p=pTSX),
            cs.daeOut(ode=ode)
            )

        self.modlT.setOption("name","T-shifted model")  
        
        
    def int_odes(self, tf, y0=None, numsteps=10000, return_endpt=False, ts=0):
        """
        This function integrates the ODEs until well past the transients. 
        This uses Casadi's simulator class, C++ wrapped in swig. Inputs:
            tf          -   the final time of integration.
            numsteps    -   the number of steps in the integration is the second argument
        """
        if y0 is None: y0 = self.y0
        
        self.integrator = cs.Integrator('cvodes',self.model)
        
        #Set up the tolerances etc.
        self.integrator.setOption("abstol", self.intoptions['int_abstol'])
        self.integrator.setOption("reltol", self.intoptions['int_reltol'])
        self.integrator.setOption("max_num_steps", self.intoptions['int_maxstepcount'])
        self.integrator.setOption("tf",tf)
        
        #Let's integrate
        self.integrator.init()
        self.ts = np.linspace(ts,tf, numsteps, endpoint=True)
        self.simulator = cs.Simulator(self.integrator, self.ts)
        self.simulator.init()
        self.simulator.setInput(y0,cs.INTEGRATOR_X0)
        self.simulator.setInput(self.param,cs.INTEGRATOR_P)
        self.simulator.evaluate()
        	
        sol = self.simulator.output().toArray().T
                
        if return_endpt==True:
            return sol[-1]
        else:
            return sol
    
    def burn_trans(self,tf=500.):
        """
        integrate the solution until tf, return only the endpoint
        """
        self.y0 = self.int_odes(tf, return_endpt=True)
        
    
    
    def solve_bvp(self, method='scipy', backup='casadi'):
        """
        Chooses between available solver methods to solve the boundary
        value problem. Backup solver invoked in case of failure
        """
        available = {
            #'periodic' : self.solveBVP_periodic,
            'casadi'   : self.solve_bvp_casadi
            ,'scipy'    : self.solve_bvp_scipy
            }

        y0in = np.array(self.y0)

        try: return available[method]()
        except Exception:
            print 'Method failed, using backup. '
            self.y0 = np.array(y0in)
            try: return available[backup]()
            except Exception:
                print 'exception2 approx y0T try'
                self.y0 = y0in
                self.approx_y0_T(tol=1E-4)
                return available[method]()
    
    def solve_bvp_scipy(self, root_method='hybr'):
        """
        Use a scipy optimize function to optimize the BVP function
        """

        # Make sure inputs are the correct format
        paramset = list(self.param)

        
        # Here we create and initialize the integrator SXFunction
        self.bvpint = cs.Integrator('cvodes',self.modlT)
        self.bvpint.setOption('abstol',self.intoptions['bvp_abstol'])
        self.bvpint.setOption('reltol',self.intoptions['bvp_reltol'])
        self.bvpint.setOption('tf',1)
        self.bvpint.setOption('disable_internal_warnings', True)
        self.bvpint.setOption('fsens_err_con', True)
        self.bvpint.init()

        def bvp_minimize_function(x):
            """ Minimization objective. X = [y0,T] """
            # perhaps penalize in try/catch?
            if all([self.intoptions['constraints']=='positive', 
                   np.any(x < 0)]): return np.ones(len(x))
            self.bvpint.setInput(x[:-1], cs.INTEGRATOR_X0)
            self.bvpint.setInput(paramset + [x[-1]], cs.INTEGRATOR_P)
            self.bvpint.evaluate()
            out = x[:-1] - self.bvpint.output().toArray().flatten()
            out = out.tolist()

            self.modlT.setInput(x[:-1], cs.DAE_X)
            self.modlT.setInput(paramset + [x[-1]], 2)
            self.modlT.evaluate()
            out += self.modlT.output()[0].toArray()[0].tolist()
            return np.array(out)
        
        from scipy.optimize import root

        options = {}

        root_out = root(bvp_minimize_function, np.append(self.y0, self.T),
                        tol=self.intoptions['bvp_ftol'],
                        method=root_method, options=options)

        # Check solve success
        if not root_out.status:
            raise RuntimeError("bvpsolve: " + root_out.message)

        # Check output convergence
        if np.linalg.norm(root_out.qtf) > self.intoptions['bvp_ftol']*1E4:
            raise RuntimeError("bvpsolve: nonconvergent")

        # save output to self.y0
        self.y0 = root_out.x[:-1]
        self.T = root_out.x[-1]
        
    def solve_bvp_casadi(self):
        """
        Uses casadi's interface to sundials to solve the boundary value
        problem using a single-shooting method with automatic differen-
        tiation.
        
        Related to PCSJ code. 
        """

        self.bvpint = cs.Integrator('cvodes',self.modlT)
        self.bvpint.setOption('abstol',self.intoptions['bvp_abstol'])
        self.bvpint.setOption('reltol',self.intoptions['bvp_reltol'])
        self.bvpint.setOption('tf',1)
        self.bvpint.setOption('disable_internal_warnings', True)
        self.bvpint.setOption('fsens_err_con', True)
        self.bvpint.init()
        
        # Vector of unknowns [y0, T]
        V = cs.MX.sym("V",self.neq+1)
        y0 = V[:-1]
        T = V[-1]
        param = cs.vertcat([self.param, T])
        yf = self.bvpint.call(cs.integratorIn(x0=y0,p=param))[0]
        fout = self.modlT.call(cs.daeIn(t=T, x=y0,p=param))[0]
        
        # objective: continuity
        obj = (yf - y0)**2  # yf and y0 are the same ..i.e. 2 ends of periodic fcn
        obj.append(fout[0]) # y0 is a peak for state 0, i.e. fout[0] is slope state 0
        
        #set up the matrix we want to solve
        F = cs.MXFunction([V],[obj])
        F.init()
        guess = np.append(self.y0,self.T)
        solver = cs.ImplicitFunction('kinsol',F)
        solver.setOption('abstol',self.intoptions['bvp_ftol'])
        solver.setOption('strategy','linesearch')
        solver.setOption('exact_jacobian', False)
        solver.setOption('pretype', 'both')
        solver.setOption('use_preconditioner', True)
        if self.intoptions['constraints']=='positive':
            solver.setOption('constraints', (2,)*(self.neq+1))
        solver.setOption('linear_solver_type', 'dense')
        solver.init()
        solver.setInput(guess)
        solver.evaluate()
        
        sol = solver.output().toArray().squeeze()
        
        self.y0 = sol[:-1]
        self.T = sol[-1]


    def dydt(self,y):
        """
        Function to calculate model for given y.
        """
        try:
            out = []
            for yi in y:
                assert len(yi) == self.neq
                self.model.setInput(yi,cs.DAE_X)
                self.model.setInput(self.param,cs.DAE_P)
                self.model.evaluate()
                out += [self.model.output().toArray().flatten()]
            return np.array(out)
        
        except (AssertionError, TypeError):
            self.model.setInput(y,cs.DAE_X)
            self.model.setInput(self.param,cs.DAE_P)
            self.model.evaluate()
            return self.model.output().toArray().flatten()

        
    def dfdp(self,y,p=None):
        """
        Function to calculate model jacobian for given y and p.
        """
        if p is None: p = self.param

        try:
            out = []
            for yi in y:
                assert len(yi) == self.neq
                self.jacp.setInput(yi,cs.DAE_X)
                self.jacp.setInput(p,cs.DAE_P)
                self.jacp.evaluate()
                out += [self.jacp.output().toArray()]
            return np.array(out)
        
        except (AssertionError, TypeError):
            self.jacp.setInput(y,cs.DAE_X)
            self.jacp.setInput(p,cs.DAE_P)
            self.jacp.evaluate()
            return self.jacp.output().toArray()

        
    def dfdy(self,y,p=None):
        """
        Function to calculate model jacobian for given y and p.
        """
        if p is None: p = self.param
        try:
            out = []
            for yi in y:
                assert len(yi) == self.neq
                self.jacy.setInput(yi,cs.DAE_X)
                self.jacy.setInput(p,cs.DAE_P)
                self.jacy.evaluate()
                out += [self.jacy.output().toArray()]
            return np.array(out)
        
        except (AssertionError, TypeError):
            self.jacy.setInput(y,cs.DAE_X)
            self.jacy.setInput(p,cs.DAE_P)
            self.jacy.evaluate()
            return self.jacy.output().toArray()

    
    def approx_y0_T(self, tout=300, burn_trans=True, tol=1E-3):
        """ 
        Approximates the period and y0 to the given tol, by integrating,
        creating a spline representation, and comparing the max values using
        state 0.        
        """
        
        if burn_trans==True:
            self.burn_trans()
        
        states = self.int_odes(tout)
        ref_state = states[:,0]
        time = self.ts
        
        # create a spline representation of the first state, k=4 so deriv k=3
        spl = UnivariateSpline(time, ref_state, k=4, s=0)
        time_spl = np.arange(0,tout,1E-3)
        
        #finds roots of splines
        roots = spl.derivative(n=1).roots() #der of spline

        # gives y0 and period by finding second deriv.        
        peaks_of_roots = np.where(spl.derivative(n=2)(roots) < 0)
        peaks = roots[peaks_of_roots]
        periods = np.diff(peaks)

        if sum(np.diff(periods)) < tol:
            self.T = np.mean(periods)
            
            #calculating the y0 for each state witha  cubic spline
            self.y0 = np.zeros(self.neq)
            for i in range(self.neq):
                spl = UnivariateSpline(time, states[:,i], k=3, s=0)
                self.y0[i] = spl(peaks[0])
                
        else:
            self.T = -1

    def corestationary(self,guess=None,contstraints='positive'):
        """
        find stationary solutions that satisfy ydot = 0 for stability
        analysis. 
        """
        guess=None
        if guess is None: guess = np.array(self.y0)
        else: guess = np.array(guess)
        y = self.model.inputExpr(cs.DAE_X)
        t = self.model.inputExpr(cs.DAE_T)
        p = self.model.inputExpr(cs.DAE_P)
        ode = self.model.outputExpr()
        fn = cs.SXFunction([y,t,p],ode)
        kfn = cs.ImplicitFunction('kinsol',fn)
        abstol = 1E-10
        kfn.setOption("abstol",abstol)
        if self.intoptions['constraints']=='positive': 
            # constain using kinsol to >0, for physical
            kfn.setOption("constraints",(2,)*self.neq) 
        kfn.setOption("linear_solver_type","dense")
        kfn.setOption("exact_jacobian",True)
        kfn.setOption("u_scale",(100/guess).tolist())
        kfn.setOption("disable_internal_warnings",True)
        kfn.init()
        kfn.setInput(self.param,2)
        kfn.setInput(guess)
        kfn.evaluate()
        y0out = kfn.output().toArray()

        if any(np.isnan(y0out)):
            raise RuntimeError("findstationary: KINSOL failed to find \
                               acceptable solution")
        
        self.ss = y0out.flatten()
        
        if np.linalg.norm(self.dydt(self.ss)) >= abstol or any(y0out <= 0):
            raise RuntimeError("findstationary: KINSOL failed to reach"+
                                " acceptable bounds")
              
        self.eigs = np.linalg.eigvals(self.dfdy(self.ss))

    def find_stationary(self, guess=None):
        """
        Find the stationary points dy/dt = 0, and check if it is a
        stable attractor (non oscillatory).
        Parameters
        ----------
        guess : (optional) iterable
            starting value for the iterative solver. If empty, uses
            current value for initial condition, y0.
        Returns
        -------
        +0 : Fixed point is not a steady-state attractor
        +1 : Fixed point IS a steady-state attractor
        -1 : Solution failed to converge
        """
        try:
            self.corestationary(guess)
            if all(np.real(self.eigs) < 0): return 1
            else: return 0

        except Exception: return -1
    
    def limit_cycle(self):
        """
        integrate the solution for one period, remembering each of time
        points along the way
        """
        
        self.ts = np.linspace(0, self.T, self.intoptions['lc_res'])
        
        intlc = cs.Integrator('cvodes',self.model)
        intlc.setOption("abstol"       , self.intoptions['lc_abstol'])
        intlc.setOption("reltol"       , self.intoptions['lc_reltol'])
        intlc.setOption("max_num_steps", self.intoptions['lc_maxnumsteps'])
        intlc.setOption("tf"           , self.T)

        intsim = cs.Simulator(intlc, self.ts)
        intsim.init()
        
        # Input Arguments
        intsim.setInput(self.y0, cs.INTEGRATOR_X0)
        intsim.setInput(self.param, cs.INTEGRATOR_P)
        intsim.evaluate()
        self.sol = intsim.output().toArray().T

        # create interpolation object
        self.lc = self.interp_sol(self.ts, self.sol.T)
        
    def interp_sol(self, tin, yin):
        """
        Function to create a periodic spline interpolater
        """
    
        return jha.MultivariatePeriodicSpline(tin, yin, period=self.T)
        
    def calc_y0(self, trans=300, bvp_method='scipy'):
        """
        meta-function to call each calculation function in order for
        unknown y0. Invoked when initial condition is unknown.
        """
        try: del self.pClass
        except AttributeError: pass
        self.burn_trans(trans)
        self.approx_y0_T(trans/3.)
        self.solve_bvp(method=bvp_method)
        #self.roots()
    
    def check_monodromy(self):
        """
        Check the stability of the limit cycle by finding the
        eigenvalues of the monodromy matrix
        """

        integrator = cs.Integrator('cvodes', self.model)
        integrator.setOption("abstol", self.intoptions['sensabstol'])
        integrator.setOption("reltol", self.intoptions['sensreltol'])
        integrator.setOption("max_num_steps", self.intoptions['int_maxstepcount'])
        integrator.setOption("sensitivity_method",
                             self.intoptions['sensmethod']);
        integrator.setOption("t0", 0)
        integrator.setOption("tf", self.T)
        integrator.setOption("fsens_err_con", 1)
        integrator.setOption("fsens_abstol", self.intoptions['sensabstol'])
        integrator.setOption("fsens_reltol", self.intoptions['sensreltol'])
        integrator.init()
        integrator.setInput(self.y0, cs.INTEGRATOR_X0)
        integrator.setInput(self.param, cs.INTEGRATOR_P)
        
        intdyfdy0 = integrator.jacobian(cs.INTEGRATOR_X0, cs.INTEGRATOR_XF)
        intdyfdy0.init()
        intdyfdy0.setInput(self.y0,"x0")
        intdyfdy0.setInput(self.param,"p")
        intdyfdy0.evaluate()
        monodromy = intdyfdy0.output().toArray()   
        
        self.monodromy = monodromy
        
        # Calculate Floquet Multipliers, check if all (besides n_0 = 1)
        # are inside unit circle
        eigs = np.linalg.eigvals(monodromy)
        self.floquet_multipliers = np.abs(eigs)
        #self.floquet_multipliers.sort()
        idx = (np.abs(self.floquet_multipliers - 1.0)).argmin()
        f = self.floquet_multipliers.tolist()
        f.pop(idx)
        
        return np.all(np.array(f) < 1)
        
    def first_order_sensitivity(self):
        """
        Function to calculate the first order period sensitivity
        matricies using the direct method. See Wilkins et al. 2009. Only
        calculates initial conditions and period sensitivities.
        """

        self.check_monodromy()
        monodromy = self.monodromy
        
        integrator = cs.Integrator('cvodes',self.model)
        integrator.setOption("abstol", self.intoptions['sensabstol'])
        integrator.setOption("reltol", self.intoptions['sensreltol'])
        integrator.setOption("max_num_steps",
                             self.intoptions['sensmaxnumsteps'])
        integrator.setOption("sensitivity_method",
                             self.intoptions['sensmethod']);
        integrator.setOption("t0", 0)
        integrator.setOption("tf", self.T)
        integrator.setOption("fsens_err_con", 1)
        integrator.setOption("fsens_abstol", self.intoptions['sensabstol'])
        integrator.setOption("fsens_reltol", self.intoptions['sensreltol'])
        integrator.init()
        integrator.setInput(self.y0,cs.INTEGRATOR_X0)
        integrator.setInput(self.param,cs.INTEGRATOR_P)
        
        intdyfdp = integrator.jacobian(cs.INTEGRATOR_P, cs.INTEGRATOR_XF)
        intdyfdp.init()
        intdyfdp.setInput(self.y0,"x0")
        intdyfdp.setInput(self.param,"p")
        intdyfdp.evaluate()
        s0 = intdyfdp.output().toArray()
        
        self.model.init()
        self.model.setInput(self.y0,cs.DAE_X)
        self.model.setInput(self.param,cs.DAE_P)
        self.model.evaluate()
        ydot0 = self.model.output().toArray().squeeze()
        
        LHS = np.zeros([(self.neq + 1), (self.neq + 1)])
        LHS[:-1,:-1] = monodromy - np.eye(len(monodromy))
        LHS[-1,:-1] = self.dfdy(self.y0)[0]
        LHS[:-1,-1] = ydot0
        
        RHS = np.zeros([(self.neq + 1), self.np])
        RHS[:-1] = -s0
        RHS[-1] = self.dfdp(self.y0)[0]
        
        unk = np.linalg.solve(LHS,RHS)
        self.S0 = unk[:-1]
        self.dTdp = unk[-1]
        self.reldTdp = self.dTdp*self.param/self.T
    
    def find_prc(self, res=100, num_cycles=20):
        """ Function to calculate the phase response curve with
        specified resolution """
    
        # Make sure the lc object exists
        if not hasattr(self, 'lc'): self.limit_cycle()
        
        # Get a state that is not at a local max/min (0 should be at
        # max)
        state_ind = 1
        while np.abs(self.dydt(self.y0)[state_ind]) < 1E-5: state_ind += 1
        
        integrator = cs.Integrator('cvodes',self.model)
        integrator.setOption("abstol", self.intoptions['sensabstol'])
        integrator.setOption("reltol", self.intoptions['sensreltol'])
        integrator.setOption("max_num_steps",
                             self.intoptions['sensmaxnumsteps'])
        integrator.setOption("sensitivity_method",
                             self.intoptions['sensmethod']);
        integrator.setOption("t0", 0)
        integrator.setOption("tf", num_cycles*self.T)
        #integrator.setOption("numeric_jacobian", True)
        integrator.setOption("fsens_err_con", 1)
        integrator.setOption("fsens_abstol", self.intoptions['sensabstol'])
        integrator.setOption("fsens_reltol", self.intoptions['sensreltol'])
        integrator.init()
        seed = np.zeros(self.neq)
        seed[state_ind] = 1.
        integrator.setInput(self.y0, cs.INTEGRATOR_X0)
        integrator.setInput(self.param, cs.INTEGRATOR_P)
        #adjseed = (seed, cs.INTEGRATOR_XF)
        integrator.evaluate()#0, 1)
        
        monodromy = integrator.jacobian(cs.INTEGRATOR_X0,cs.INTEGRATOR_XF)
        monodromy.init()
        monodromy.setInput(self.y0,"x0")
        monodromy.setInput(self.param,"p")
        monodromy.evaluate()
        # initial state is Kcross(T,T) = I
        adjsens = monodromy.getOutput().toArray().T.dot(seed)
        
        from scipy.integrate import odeint
        def adj_func(y, t):
            """ t will increase, trace limit cycle backwards through -t. y
            is the vector of adjoint sensitivities """
            jac = self.dfdy(self.lc((-t)%self.T))
            return y.dot(jac)
            
        seed = adjsens
        self.prc_ts = np.linspace(0, self.T, res)
        P = odeint(adj_func, seed, self.prc_ts)[::-1] # Adjoint matrix at t 

        self.sPRC = self._t_to_phi(P/self.dydt(self.y0)[state_ind])
        
        dfdp = np.array([self.dfdp(self.lc(t)) for t in self.prc_ts])
        # Must rescale f to \hat{f}, inverse of rescaling t
        self.pPRC = self._t_to_phi(
                        np.array([self.sPRC[i].dot(self._phi_to_t(dfdp[i]))
                              for i in xrange(len(self.sPRC))])
                                  )
        self.rel_pPRC = self.pPRC*np.array(self.param)

        # Create interpolation object for the state phase response curve
        self.sPRC_interp = self.interp_sol(self.prc_ts, self.sPRC.T) #phi units
        self.pPRC_interp = self.interp_sol(self.prc_ts, self.pPRC.T) #phi units
        
    def _create_ARC_model(self, numstates=1):
        """ Create model with quadrature for amplitude sensitivities
        numstates might allow us to calculate entire sARC at once, but
        now will use seed method. """

        # Allocate symbolic vectors for the model
        dphidx = cs.SX.sym('dphidx', numstates)
        t      = self.model.inputExpr(cs.DAE_T)    # time
        xd     = self.model.inputExpr(cs.DAE_X)    # differential state
        s      = cs.SX.sym("s", self.neq, numstates) # sensitivities
        p      = cs.vertcat([self.model.inputExpr(2), dphidx]) # parameters

        # Symbolic function (from input model)
        ode_rhs = self.model.outputExpr()[0]
        f_tilde = self.T*ode_rhs/(2*np.pi)

        # symbolic jacobians
        jac_x = self.model.jac(cs.DAE_X, cs.DAE_X)   
        sens_rhs = jac_x.mul(s)

        quad = cs.SX.sym('q', self.neq, numstates)
        for i in xrange(numstates):
            quad[:,i] = 2*(s[:,i] - dphidx[i]*f_tilde)*(xd - self.avg)

        shape = (self.neq*numstates, 1)

        x = cs.vertcat([xd, s.reshape(shape)])
        ode = cs.vertcat([ode_rhs, sens_rhs.reshape(shape)])
        ffcn = cs.SXFunction(cs.daeIn(t=t, x=x, p=p),
                             cs.daeOut(ode=ode, quad=quad))
        return ffcn


    def _sarc_single_time(self, time, seed):
        """ Calculate the state amplitude response to an infinitesimal
        perturbation in the direction of seed, at specified time. """

        # Initialize model and sensitivity states
        x0 = np.zeros(2*self.neq)
        x0[:self.neq] = self.lc(time)
        x0[self.neq:] = seed
        
        # Add dphi/dt from seed perturbation
        param = np.zeros(self.np + 1)
        param[:self.np] = self.param
        param[-1] = self.sPRC_interp(time).dot(seed)

        # Evaluate model
        self.sarc_int.setInput(x0, cs.INTEGRATOR_X0)
        self.sarc_int.setInput(param, cs.INTEGRATOR_P)
        self.sarc_int.evaluate()
        amp_change = self.sarc_int.output(cs.INTEGRATOR_QF).toArray()
        self.sarc_int.reset()

        amp_change *= (2*np.pi)/(self.T)

        return amp_change


    def _findARC_seed(self, seeds, res=100, trans=3): 

        # Calculate necessary quantities
        if not hasattr(self, 'avg'): self.average()
        if not hasattr(self, 'sPRC'): self.find_prc(res)

        # Set up quadrature integrator
        self.sarc_int = cs.Integrator('cvodes',self._create_ARC_model())
        self.sarc_int.setOption("abstol", self.intoptions['sensabstol'])
        self.sarc_int.setOption("reltol", self.intoptions['sensreltol'])
        self.sarc_int.setOption("max_num_steps",
                             self.intoptions['sensmaxnumsteps'])
        self.sarc_int.setOption("t0", 0)
        self.sarc_int.setOption("tf", trans*self.T)
        #self.sarc_int.setOption("numeric_jacobian", True)

        self.sarc_int.init()

        t_arc = np.linspace(0, self.yT, res)
        arc = np.array([self._sarc_single_time(t, seed) for t, seed in
                        zip(t_arc, seeds)]).squeeze()
        return t_arc, arc

    def findSARC(self, state, res=100, trans=3):
        """ Find amplitude response curve from pertubation to state """
        seed = np.zeros(self.neq)
        seed[state] = 1.
        return self._findARC_seed([seed]*res, res, trans)

    def findPARC(self, param, res=100, trans=3, rel=False):
        """ Find ARC from temporary perturbation to parameter value """
        t_arc = np.linspace(0, self.T, res)
        dfdp = self._phi_to_t(self.dfdp(self.lc(t_arc))[:,:,param])
        t_arc, arc = self._findARC_seed(dfdp, res, trans)
        if rel: arc *= self.param[param]/self.avg
        return t_arc, arc

    def findARC_whole(self, res=100, trans=3):
        """ Calculate entire sARC matrix, which will be faster than
        calcualting for each parameter """

        # Calculate necessary quantities
        if not hasattr(self, 'avg'): self.average()
        if not hasattr(self, 'sPRC'): self.find_prc(res)

        # Set up quadrature integrator
        self.sarc_int = cs.Integrator('cvodes',
            self._create_ARC_model(numstates=self.neq))
        self.sarc_int.setOption("abstol", self.intoptions['sensabstol'])
        self.sarc_int.setOption("reltol", self.intoptions['sensreltol'])
        self.sarc_int.setOption("max_num_steps",
                             self.intoptions['sensmaxnumsteps'])
        self.sarc_int.setOption("t0", 0)
        self.sarc_int.setOption("tf", trans*self.T)
        #self.sarc_int.setOption("numeric_jacobian", True)
        self.sarc_int.init()

        self.arc_ts = np.linspace(0, self.T, res)
        
        amp_change = []
        for t in self.arc_ts:
            # Initialize model and sensitivity states
            x0 = np.zeros(self.neq*(self.neq + 1))
            x0[:self.neq] = self.lc(t)
            x0[self.neq:] = np.eye(self.neq).flatten()
            
            # Add dphi/dt from seed perturbation
            param = np.zeros(self.np + self.neq)
            param[:self.np] = self.param
            param[self.np:] = self.sPRC_interp(t)

            # Evaluate model
            self.sarc_int.setInput(x0, cs.INTEGRATOR_X0)
            self.sarc_int.setInput(param, cs.INTEGRATOR_P)
            self.sarc_int.evaluate()
            out = self.sarc_int.output(cs.INTEGRATOR_QF).toArray()
            # amp_change += [out]
            amp_change += [out*2*np.pi/self.T]
                                        

        #[time, state_out, state_in]
        self.sARC = np.array(amp_change)
        dfdp = np.array([self.dfdp(self.lc(t)) for t in self.arc_ts])
        self.pARC = np.array([self.sARC[i].dot(self._phi_to_t(dfdp[i]))
                              for i in xrange(len(self.sARC))])

        self.rel_pARC = (np.array(self.param) * self.pARC /
                         np.atleast_2d(self.avg).T)        
        
    def _cos_components(self):
        """ return the phases and amplitudes associated with the first
        order fourier compenent of the limit cycle (i.e., the best-fit
        sinusoid which fits the limit cycle) """
    
        if not hasattr(self, 'sol'): self.limit_cycle()
        
        dft_sol = np.fft.fft(self.sol[:], axis=0)
        n = len(self.ts[:-1])
        baseline = dft_sol[0]/n
        comp = 2./n*dft_sol[1]
        return np.abs(comp), np.angle(comp), baseline
        
    def average(self):
        """
        integrate the solution with quadrature to find the average 
        species concentration. outputs to self.avg
        """
        
        ffcn_in = self.model.inputExpr()
        ode = self.model.outputExpr()
        quad = cs.vertcat([ffcn_in[cs.DAE_X], ffcn_in[cs.DAE_X]**2])

        quadmodel = cs.SXFunction(ffcn_in, cs.daeOut(ode=ode[0], quad=quad))

        qint = cs.Integrator('cvodes',quadmodel)
        qint.setOption("abstol"        , self.intoptions['lc_abstol'])
        qint.setOption("reltol"        , self.intoptions['lc_reltol'])
        qint.setOption("max_num_steps" , self.intoptions['lc_maxnumsteps'])
        qint.setOption("tf",self.T)
        qint.init()
        qint.setInput(self.y0, cs.INTEGRATOR_X0)
        qint.setInput(self.param, cs.INTEGRATOR_P)
        qint.evaluate()
        quad_out = qint.output(cs.INTEGRATOR_QF).toArray().squeeze()
        self.avg = quad_out[:self.neq]/self.T
        self.rms = np.sqrt(quad_out[self.neq:]/self.T)
        self.std = np.sqrt(self.rms**2 - self.avg**2)

    def lc_phi(self, phi):
        """ interpolate the selc.lc interpolation object using a time on
        (0,2*pi) """
        return self.lc(self._phi_to_t(phi%(2*np.pi)))
        
    def phase_of_point(self, point, error=False, tol=1E-3):
        """ Finds the phase at which the distance from the point to the
        limit cycle is minimized. phi=0 corresponds to the definition of
        y0, returns the phase and the minimum distance to the limit
        cycle """
        
            
        point = np.asarray(point)
        
        #set up integrator so we only have to once...
        intr = cs.Integrator('cvodes',self.model)
        intr.setOption("abstol", self.intoptions['bvp_abstol'])
        intr.setOption("reltol", self.intoptions['bvp_reltol'])
        intr.setOption("tf", self.T)
        intr.setOption("max_num_steps",
                       self.intoptions['transmaxnumsteps'])
        intr.setOption("disable_internal_warnings", True)
        intr.init()
        for i in xrange(100):
            dist = cs.SX.sym("dist")
            x = self.model.inputExpr(cs.DAE_X)
            ode = self.model.outputExpr()[0]
            dist_ode = cs.sumAll(2.*(x - point)*ode)

            cat_x   = cs.vertcat([x, dist])
            cat_ode = cs.vertcat([ode, dist_ode])

            dist_model = cs.SXFunction(
                cs.daeIn(t=self.model.inputExpr(cs.DAE_T), x=cat_x,
                         p=self.model.inputExpr(cs.DAE_P)),
                cs.daeOut(ode=cat_ode))

            dist_model.setOption("name","distance model")

            dist_0 = ((self.y0 - point)**2).sum()
            if dist_0 < tol:
                # catch the case where we start at 0
                return 0.
            cat_y0 = np.hstack([self.y0, dist_0])

            roots_class = Oscillator(dist_model, self.param, cat_y0)
            #return roots_class
            roots_class.approx_y0_T()
            roots_class.solve_bvp()
            roots_class.limit_cycle()
            roots_class.roots()

            phases = self._t_to_phi(roots_class.tmin[-1])
            distances = roots_class.ymin[-1]
            phase_ind = np.argmin(distances)
            found_phase = phases[phase_ind] 
            # distance min is in correct location but the initial condition
            # is sensitive, so we re-calculate the distance here.
            distance = np.sum(
                (roots_class.lc(self._phi_to_t(found_phase))[:-1]-point)**2)

            if all([distance < tol, distance > 0]): 
                # for multiple minima
                return phases[phase_ind]#, roots_class

            intr.setInput(point, cs.INTEGRATOR_X0)
            intr.setInput(self.param, cs.INTEGRATOR_P)
            intr.evaluate()
            point = intr.output().toArray().flatten() #advance by one cycle
        raise RuntimeError("Point failed to converge to limit cycle")
    
    def roots(self, res=500):
        """
        Mediocre reproduction of Peter's roots fcn. Returns full max/min
        values and times for each state in the system. This is performed with
        splines. Warning: splines can be messy near discontinuities or highly-
        nonlinear regions.
        """
        if not self.lc:
            self.limit_cycle()
        tin = np.linspace(0, self.T, res)
        yin = self.lc(tin)
        
        # get splines of der1 and der2 for finding roots
        # make initial LC spline k so that we can find roots at all levels
        sps = self.lc.splines
        
        sp4 = jha.MultivariatePeriodicSpline(tin, yin.T, period=self.T, k=4)
        der1s = [spi.derivative(n=1) for spi in sp4.splines]
        
        sp5 = jha.MultivariatePeriodicSpline(tin, yin.T, period=self.T, k=5)
        der2s = [spi.derivative(n=2) for spi in sp5.splines]

        self.tmax = np.array([d1.roots()[der2s[i](d1.roots()) < 0] 
                        for i,d1 in enumerate(der1s)])
        self.ymax = np.array([spi(self.tmax[i]) for i,spi in enumerate(sps)])

        self.tmin = np.array([d1.roots()[der2s[i](d1.roots()) > 0] 
                        for i,d1 in enumerate(der1s)])
        self.ymin = np.array([spi(self.tmin[i]) for i,spi in enumerate(sps)])


if __name__ == "__main__":
    
    from Models.tyson_model import model, param, EqCount
    
    lap = jha.laptimer()
    
    # initialize with 1s
    tyson = Oscillator(model(), param, np.ones(EqCount))
    print tyson.y0, 'setup time = %0.3f' %(lap() )
    
    # find a spot on LC
    tyson.y0 = np.ones(EqCount)
    tyson.burn_trans()
    print tyson.y0, 'y0 burn time = %0.3f' %(lap() ) 
    
    # or find a max and the approximate period
    tyson.approx_y0_T()
    print tyson.y0, tyson.T, 'y0 approx time = %0.3f' %(lap() )
    
    # find the period and y0 using a BVP solution
    tyson.solve_bvp(method='scipy')
    print tyson.y0, tyson.T, 'y0 scipy bvp time = %0.3f' %(lap() )
    
    # find the period and y0 using a BVP solution
    tyson.solve_bvp(method='casadi')
    print tyson.y0, tyson.T, 'y0 casadi bvp time = %0.3f' %(lap() )

    # find steady state soln (fixed pt)
    tyson.find_stationary()
    print tyson.ss, 'stationary time = %0.3f' %(lap() )
        # or perform everything all at once
    tyson = Oscillator(model(), param, np.ones(EqCount))
    tyson.calc_y0()
    print tyson.y0, tyson.T, 'y0 start-finish = %0.3f' %(lap() )
    
    # adiitional analysis tools
    tyson.limit_cycle()
    tyson.first_order_sensitivity()
    intg = tyson.find_prc()
    
    



