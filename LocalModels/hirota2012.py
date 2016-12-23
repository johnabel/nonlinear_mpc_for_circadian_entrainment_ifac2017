# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:10:20 2014

@author: john
"""

import numpy  as np
import casadi as cs

EqCount = 8
ParamCount = 21

y0in = np.array([1.33593236, 1.79197778, 1.62225403, 1.62164561,
                 5.17596319, 3.96326956, 0.37392169, 0.1865519])
                 
period = 23.69999994

param = np.array([1.94599211e-01 , 1.30641511e-01 , 1.13505694e-01 ,
                    4.25460804e-01 , 2.59493541e-01 , 3.26290677e-01 ,
                    6.76063523e-01 , 6.07937881e-01 , 1.14786910e-02 ,
                    1.14911580e+00 , 2.96993795e+00 , 3.38240851e-02 ,
                    1.52309394e+00 , 1.68558083e+00 , 2.01721235e+00 ,
                    1.01201779e-01 , 3.31792942e+00 , 5.26256372e-02 ,
                    4.06306642e-02 , 1.75478279e-03 , 3.00000000e+00])

def model():
    """ trying non-reversible nuclear entry and fitting knockouts. """
    #==================================================================
    #setup of symbolics
    #==================================================================
    p   = cs.SX.sym("p")
    c1  = cs.SX.sym("c1")
    c2  = cs.SX.sym("c2")
    P   = cs.SX.sym("P")
    C1  = cs.SX.sym("C1")
    C2  = cs.SX.sym("C2")
    C1P = cs.SX.sym("C1P")
    C2P = cs.SX.sym("C2P")
    
    y = cs.vertcat([p, c1, c2, P, C1, C2, C1P, C2P])
    
    # Time Variable
    t = cs.SX.sym("t")
    
    
    #===================================================================
    #Parameter definitions
    #===================================================================
    vtp    = cs.SX.sym("vtp")
    vtc1   = cs.SX.sym("vtc1")
    vtc2   = cs.SX.sym("vtc2")
    knp    = cs.SX.sym("knp")
    knc1   = cs.SX.sym("knc1")
    vdp    = cs.SX.sym("vdp")
    vdc1   = cs.SX.sym("vdc1")
    vdc2   = cs.SX.sym("vdc2")
    kdp    = cs.SX.sym("kdp")
    kdc1   = cs.SX.sym("kdc1")
    vdP    = cs.SX.sym("vdP")
    vdC1   = cs.SX.sym("vdC1")
    vdC2   = cs.SX.sym("vdC2")
    vdCn   = cs.SX.sym("vdCn")
    MC2n   = cs.SX.sym("MC2n")
    kdP    = cs.SX.sym("kdP")
    kdC1   = cs.SX.sym("kdC1")
    kdCn   = cs.SX.sym("kdCn")
    vaC1P  = cs.SX.sym("vaC1P")
    vdC1P  = cs.SX.sym('vdC1P')
    ktxnp  = cs.SX.sym('ktxnp')


    paramset = cs.vertcat([vtp  , vtc1 , vtc2  , knp   , knc1   , vdp  ,
                        vdc1 , vdc2  , kdp  , kdc1 , vdP   , kdP   ,
                        vdC1   , vdC2 , kdC1 , vdCn , MC2n , kdCn ,
                        vaC1P , vdC1P , ktxnp])
    
    # Model Equations
    ode = [[]]*EqCount
    
    def txn(vmax,km,np,dact1,dact2):
        return vmax/(km + (dact1 + dact2)**np)
    
    def txl(mrna,kt):
        return kt*mrna
    
    def MMdeg(species,vmax,km):
        return -vmax*(species)/(km+species)
        
    def cxMMdeg(species1,species2,vmax,km):
        return -vmax*(species1)/(km + species1 + species2)
        
    def cx(s1,s2,cmplx,ka,kd):
        # positive for reacting species, negative for complex
        return -ka*s1*s2 + kd*cmplx
        
    # MRNA Species
    ode[0] = txn(vtp,knp,3,C1P,C2P)   + MMdeg(p,vdp,kdp)
    ode[1] = txn(vtc1,knc1,3,C1P,C2P) + MMdeg(c1,vdc1,kdc1)
    ode[2] = txn(vtc2,knc1,3,C1P,C2P) + MMdeg(c2,vdc2,kdc1)
    
    # Free Proteins
    ode[3] = txl(p,ktxnp) + MMdeg(P,vdP,kdP)    + cx(P,C1,C1P,vaC1P,vdC1P)\
                                                + cx(P,C2,C2P,vaC1P,vdC1P)
    ode[4] = txl(c1,1)    + MMdeg(C1,vdC1,kdC1) + cx(P,C1,C1P,vaC1P,vdC1P)
    ode[5] = txl(c2,1)    + MMdeg(C2,vdC2,kdC1) + cx(P,C2,C2P,vaC1P,vdC1P)

    # PER/CRY Cytoplasm Complexes
    ode[6] = -cx(P,C1,C1P,vaC1P,vdC1P) + cxMMdeg(C1P,C2P,vdCn,kdCn)
    ode[7] = -cx(P,C2,C2P,vaC1P,vdC1P) + cxMMdeg(C2P,C1P,vdCn*MC2n,kdCn)

    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=paramset),
                       cs.daeOut(ode=ode))
    
    fn.setOption("name","degmodel")
    
    return fn