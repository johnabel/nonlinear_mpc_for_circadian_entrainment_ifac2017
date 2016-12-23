"""
A mixed file containing some utilities that I find useful in solving 
circadian problems.

jha
"""

#import modules
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import (splrep, splint, fitpack, splev,
                               UnivariateSpline, dfitpack,
                               InterpolatedUnivariateSpline)
import matplotlib.pyplot as plt
from time import time
import pdb

def roots(data,times=None):
    """
    Takes a set of data and finds the roots of it. Uses a spline 
    interpolation to get the root values.
    """

    if times is None:
        #time intervals set to one
        times = np.arange(len(data))

    #fits a spline centered on those indexes
    s = UnivariateSpline(times, data, s=0)

    return s.roots()

class laptimer:
    """
    Whenever you call it, it times laps.
    """
    
    def __init__(self):
        self.time = time()

    def __call__(self):
        ret = time() - self.time
        self.time = time()
        return ret

    def __str__(self):
        return "%.3E"%self()

    def __repr__(self):
        return "%.3E"%self()

class spline:
    """ Periodic data interpolation object used by Collocation. Probably
    could stand an update """
    def __init__(self,tvals,yvals,sfactor):
        self.max = np.array(yvals).max()
        self.min = np.array(yvals).min()
        self.amp = self.max-self.min
        # scaled y (0->1)
        self.yscaled = (yvals - self.min)/self.amp
        smooth = sfactor*(len(tvals) - np.sqrt(2*len(tvals)))
        spl = splrep(tvals,self.yscaled,s=smooth,per=True)
        self.spl = spl

    def __call__(self,s,d=0):
        if d == 0:
            return self.amp*(splev(s, self.spl, der=d)) + self.min
        else:
            return self.amp*(splev(s, self.spl, der=d))

class fnlist(list):
    def __call__(self,*args,**kwargs):
        return np.array([entry(*args,**kwargs) for entry in self])

def corrsort(mat):
    """ Function to sort correlation matrix so that correlated variables
    are closer to eachother.  """

    # Get the eigenvalues and eigenvectors, sort them for increasing
    # order
    w, v = np.linalg.eig(mat)
    v = v[:,w.argsort()]
    w.sort()

    e1 = v[:,-1]
    e2 = v[:,-2]

    angles = np.arctan(e2/e1) + ~(e1 > 0) * np.pi
    order = angles.argsort()
    angles = angles[order]
    maxdiff = np.diff(angles).argmax() + 1

    # Expand at maximum angular difference
    order = np.concatenate((order[maxdiff:],order[:maxdiff]))

    # reorder matrix rows, columns
    mat = mat[order][:,order]

    return mat, order


def bode(G,f=np.arange(.01,100,.01),desc=None,color=None):

    jw = 2*np.pi*f*1j
    y = np.polyval(G.num, jw) / np.polyval(G.den, jw)
    mag = 20.0*np.log10(abs(y))
    phase = np.arctan2(y.imag, y.real)*180.0/np.pi % 360

    #plt.semilogx(jw.imag, mag)
    plt.semilogx(f,mag,label=desc,color=color)

    return mag, phase



     

class PeriodicSpline(UnivariateSpline):
    def __init__(self, x, y, period=2*np.pi, sfactor=0, k=3, ext=0):
        """ 
        A PCSJ spline class
        Function to define a periodic spline that approximates a
        continous function sampled by x and y data points. If the repeat
        data point is not provided, it will be added to ensure a
        periodic trajectory """

        # Process inputs
        assert len(x) == len(y), "Length Mismatch"
        assert x.ndim == 1 & y.ndim == 1, "Too many dimensions"
        if not np.abs(x[-1] - period) < 1E-10:
            assert x[-1] < period, 'Data longer than 1 period'
            x = np.hstack([x, x[0]+period])
            y = np.hstack([y, y[0]])
        

        self.T = period
        self.ext = ext

        tck = splrep(x, y, s=sfactor, per=True, k=k) 
        t, c, k = tck
        self._eval_args = tck
        self._data = (None,None,None,None,None,k,None,len(t),t,
                      c,None,None,None,None)

    def __call__(self, x, nu=0):
        return UnivariateSpline.__call__(self, x%self.T, nu=nu)

    def derivative(self, n=1):
        tck = fitpack.splder(self._eval_args, n)
        return PeriodicSpline._from_tck(tck, self.T)

    def antiderivative(self, n=1):
        tck = fitpack.splantider(self._eval_args, n)
        return PeriodicSpline._from_tck(tck)

    def root_offset(self, root=0):
        """ Return the values where the spline equals 'root'
        Restriction: only cubic splines are supported by fitpack.
        """
        t, c, k = self._eval_args
        new_c = np.array(c)
        new_c[np.nonzero(new_c)] += -root
        if k == 3:
            z,m,ier = dfitpack.sproot(t, new_c)
            if not ier == 0:
                raise ValueError("Error code returned by spalde: %s" % ier)
            return z[:m]
        raise NotImplementedError('finding roots unsupported for '
                                    'non-cubic splines')

    def integrate(self, a=0., b=2*np.pi):
        """ Find the definite integral of the spline from a to b """

        # Are both a and b in (0, 2pi)?
        if (0 <= a <= 2*np.pi) and (0 <= b <= 2*np.pi):
            return splint(a, b, self._eval_args)
        elif ((a <= 0) and (b <= 0)) or ((a >= 2*np.pi) 
                                         and (b >= 2*np.pi)):
            return splint(a%(2*np.pi), b%(2*np.pi), self._eval_args)

        elif (a <= 0) or (b >= 2*np.pi):
            int = 0
            int += splint(a%(2*np.pi), 2*np.pi, self._eval_args)
            int += splint(0, b%(2*np.pi), self._eval_args)
            return int


    @classmethod
    def _from_tck(cls, tck, period=2*np.pi):
        """Construct a spline object from given tck"""
        self = cls.__new__(cls)
        self.T = period
        t, c, k = tck
        self._eval_args = tck
        #_data == x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier
        self._data = (None,None,None,None,None,k,None,len(t),t,
                      c,None,None,None,None)
        self.ext = 0
        return self



class ComplexPeriodicSpline:
    def __init__(self, x, y, period=2*np.pi, sfactor=0):
        """
        A PCSJ spline class
        Class for complex periodic functions that will create two
        PeriodicSpline instances, one for real and one for imaginary
        components """

        yreal = np.real(y)
        yimag = np.imag(y)

        self.real_interp = PeriodicSpline(x, yreal, period, sfactor)
        self.imag_interp = PeriodicSpline(x, yimag, period, sfactor)
    
    def __call__(self, x, d=0):
        return self.real_interp(x, d) + 1j*self.imag_interp(x, d)

    def integrate(self, a, b):
        return (self.real_interp.integrate(a, b) + 
                self.imag_interp.integrate(a, b)*1j)


class MultivariatePeriodicSpline(object):
    def __init__(self, x, ys, period=2*np.pi, sfactor=0, k=3):
        """ 
        A PCSJ spline class
        Combination class that supports a multi-dimensional input,
        will determine whether complex or regular periodic splines are
        needed. """

        self.iscomplex = np.any(np.iscomplex(ys))
        splinefn = (ComplexPeriodicSpline if self.iscomplex else
                    PeriodicSpline)

        self.splines = fnlist([])
        for y in np.atleast_2d(ys):
            y = y.squeeze()
            self.splines += [splinefn(x, y, period, sfactor, k)]

    def __call__(self, x, d=0):
        return self.splines(x, d).T

    def integrate(self, a=0, b=2*np.pi):
        return np.array([interp.integrate(a,b) for interp in
                         self.splines])

def p_integrate(x, y, meth='spline'):
    """ Integrate y(x), assuming x in (0, 2*pi). x and y should contain
    the last point (x[-1] = 2*pi, y[-1] = y[0]). y[i,j,k] can be
    multi-dimensional, the first axis i should correspond to the length
    of x, and is the axis of integration. The function will return
    z[j,k], a matrix of integrated values. """

    if y.ndim < 2: y = np.atleast_2d(y).T
    assert len(x) == y.shape[0], "Shape mismatch"

    try: return _p_integrate(x, y, meth)
    except ValueError:
        return np.array([_p_integrate(x, yi.T, meth) for yi in y.T])

def ptc_from_prc(prc):
    """ Function to return a callable function (x, d=0) to interpolate
    the phase transition curve and supply derivative information """

    def ptc_interp(x, n=0):
        if n==0: return x + prc(x)
        elif n==1: return 1. + prc(x, 1)
        elif n>=2: return prc(x, n)
        else: raise RuntimeError("Negative values of d not allowed")

    return ptc_interp
    
def _p_integrate(x, y, meth):


    methods = {
        'spline' : spline_periodic_integration,
        'trapz'   : trapezoid_periodic_integration,
        'sum'    : sum_periodic_integration,
    }

    if meth == 'spline' and np.any(np.iscomplex(y)):
        real = methods['spline'](x, np.real(y))
        imag = methods['spline'](x, np.imag(y))
        ret = real + imag*1j

    else: ret = methods[meth](x, y)

    return ret.squeeze()

def sum_periodic_integration(x, y):
    return 2*np.pi*y[:-1].sum(0)/(y.shape[-1])
        
def trapezoid_periodic_integration(x, y):
    return np.trapz(y, x=x, axis=0)

def spline_periodic_integration(x, y):
    """ Quickly integrates y(x) from x=(0, 2*np.pi). x must be evenly
    spaced and include the endpoint (y[-1] = y[0]). y should be matrix
    with shape [j, i], j = len(x) """

    ii = y.shape[1]
    dx = x[1]
    tck0 = np.hstack([-3*dx, -2*dx, -dx, x,
                      [x[-1] + i*dx for i in xrange(1,4)]])
    a = [y[-2][:,None], y.T, y[1][:,None], np.zeros((ii,4))]
    tck1_arr = np.hstack(a)

    out = np.array([splint(0, 2*np.pi, (tck0, tck1, 3)) for tck1 in
                    tck1_arr])

    return out    

class RootFindingSpline(InterpolatedUnivariateSpline):
    def root_offset(self, root=0):
        """ Return the values where the spline equals 'root'
        Restriction: only cubic splines are supported by fitpack.
        """
        t, c, k = self._eval_args
        new_c = np.array(c)
        new_c[np.nonzero(new_c)] += -root
        if k == 3:
            z,m,ier = dfitpack.sproot(t, new_c)
            if not ier == 0:
                raise ValueError("Error code returned by spalde: %s" % ier)
            return z[:m]
        raise NotImplementedError('finding roots unsupported for '
                                    'non-cubic splines')

if __name__ == "__main__":

    #test roots
    times = np.arange(0,10,0.1)
    xvals = np.sin(times)
    sine_roots = roots(xvals, times=times)
    print 'The roots of sine are:'
    print sine_roots
    print 'Root finding successful.'
