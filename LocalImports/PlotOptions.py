# Bunch of overrides for matplotlib settings not changeable in
# matplotlibrc. To use, 
#
# PCSJ
#
# >>> from CommonFiles.PlotOptions import PlotOptions, layout_pad
# >>> PlotOptions()
# >>> *** make plots here ***
# >>> fig.tight_layout(**layout_pad)

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib



def PlotOptions(uselatex=False, ticks='out'):

    import matplotlib
    import matplotlib.axis, matplotlib.scale 
    from matplotlib.ticker import (MaxNLocator, NullLocator,
                                   NullFormatter, ScalarFormatter)

    MaxNLocator.default_params['nbins']=6
    MaxNLocator.default_params['steps']=[1, 2, 5, 10]

    def set_my_locators_and_formatters(self, axis):
        # choose the default locator and additional parameters
        if isinstance(axis, matplotlib.axis.XAxis):
            axis.set_major_locator(MaxNLocator())
        elif isinstance(axis, matplotlib.axis.YAxis):
            axis.set_major_locator(MaxNLocator())
        # copy&paste from the original method
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    #override original method
    matplotlib.scale.LinearScale.set_default_locators_and_formatters = \
            set_my_locators_and_formatters

    matplotlib.backend_bases.GraphicsContextBase.dashd = {
            'solid': (None, None),
            'dashed': (0, (2.0, 2.0)),
            'dashdot': (0, (1.5, 2.5, 0.5, 2.5)),
            'dotted': (0, (0.25, 1.50)),
        }
        
    # colors: 377EB8, E41A1C, 4DAF4A, 984EA3, FF7F00, FFFF33, A65628, F781BF
    #medium darkness
    matplotlib.colors.ColorConverter.colors['f'] = \
            (55/255, 126/255, 184/255)
    matplotlib.colors.ColorConverter.colors['h'] = \
            (228/255, 26/255, 28/255)
    matplotlib.colors.ColorConverter.colors['i'] = \
            (77/255, 175/255, 74/255)
    matplotlib.colors.ColorConverter.colors['j'] = \
            (152/255, 78/255, 163/255)
    matplotlib.colors.ColorConverter.colors['l'] = \
            (255/255, 127/255, 0/255)
    
    #lighter
    matplotlib.colors.ColorConverter.colors['fl'] = \
            (166/255, 206/255,227/255)
    matplotlib.colors.ColorConverter.colors['hl'] = \
            (251/255, 154/255, 153/255)
    matplotlib.colors.ColorConverter.colors['il'] = \
            (178/255, 223/255,138/255)
    matplotlib.colors.ColorConverter.colors['jl'] = \
            (202/255, 178/255, 214/255)
    matplotlib.colors.ColorConverter.colors['ll'] = \
            (253/255, 191/255, 111/255)
            
    if uselatex:
        matplotlib.rc('text', usetex=True)
        matplotlib.rc('font', family='serif')
    
    from matplotlib import rcParams
    rcParams['xtick.direction'] = ticks
    rcParams['ytick.direction'] = ticks
            
        

# Padding for formatting figures using tight_layout
layout_pad = {
    'pad'   : 0.05,
    'h_pad' : 0.6,
    'w_pad' : 0.6}

# Plot shortcuts for a number of circadian-relevant features

def plot_gray_zero(ax, **kwargs):
    ax.axhline(0, ls='--', color='grey', **kwargs)

def format_2pi_axis(ax, x=True, y=False):
    import numpy as np
    if x:
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xlim([0, 2*np.pi])
        ax.set_xticklabels(['$0$', r'$\nicefrac{\pi}{2}$', r'$\pi$',
                            r'$\nicefrac{3\pi}{2}$', r'$2\pi$'])
    if y:
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_ylim([-np.pi, np.pi])
        ax.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$','$0$',
                            r'$\frac{\pi}{2}$', r'$\pi$'])


def format_4pi_axis(ax, x=True, y=False):
    import numpy as np
    if x:
        ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
        ax.set_xlim([0, 4*np.pi])
        ax.set_xticklabels(['$0$', r'$\pi$', r'$2\pi$',
                            r'$3\pi$', r'$4\pi$'])
    if y:
        ax.set_yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
        ax.set_ylim([-2*np.pi, 2*np.pi])
        ax.set_yticklabels([r'-2$\pi$', r'-$\pi$','0',
                            r'$\pi$', r'2$\pi$'])

def format_npi_axis(ax, n=6, x=True, y=False):
    import numpy as np
    if x:
        ax.set_xticks([nn*np.pi for nn in range(n+1)])
        ax.set_xlim([0, n*np.pi])
        ax.set_xticklabels(['$0$', r'$\pi$', r'$2\pi$',
                            r'$3\pi$', r'$4\pi$', '$5\pi$', r'$6\pi$'])
    if y:
        pass
    
# def highlight_xrange(ax, xmin, xmax, color='y', alpha=0.5, **kwargs):
#     ax.axvspan(xmin, xmax, color=color, alpha=alpha, **kwargs)

class HistRCToggle:
    """ Class to toggle the xtick directional update of
    histogram-specific RC settings """

    hist_params = {'xtick.direction' : 'out',
                   'ytick.direction' : 'out'}

    def __init__(self):
        self.rcdef = plt.rcParams.copy()

    def on(self):
        plt.rcParams.update(self.hist_params)

    def off(self):
        plt.rcParams.update(self.rcdef)


blue = '#9999ff'
red = '#ff9999'
'''
def histogram(ax, data1=None, data2=None, color1=blue, color2=red,
              bins=20, range=None, alpha=1., label1=None, label2=None):
    """ Function to display a pretty histogram of up to two different
    distributions. Approximates transparency to get around annoying
    issue of pdflatex and matplotlib. """
    
    from mimic_alpha import colorAlpha_to_rgb
    weights1 = np.ones_like(data1)/len(data1)

    hist1 = ax.hist(data1, range=range, bins=bins, weights=weights1,
                    facecolor=color1, edgecolor='white', label=label1)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('axes', -0.05))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('axes', -0.05))
    if range:
        ax.set_xlim([range[0]*(1-1E-3), range[1]*(1+1E-3)])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    if data2 is not None:
        weights2 = np.ones_like(data2)/len(data2)
        c2_on_w = colorAlpha_to_rgb(color2, alpha=0.5, bg='w')[0]
        c2_on_c1 = colorAlpha_to_rgb(color2, alpha=0.5, bg=color1)[0]

        hist2 = ax.hist(data2, range=range, bins=bins, weights=weights2,
                        facecolor=c2_on_w, edgecolor='white',
                        label=label2)

        ax.legend(loc='upper left')

        orders = hist2[0] > hist1[0]
        for i, order in enumerate(orders):
            if order:
                hist1[-1][i].set_facecolor(c2_on_c1)
                hist1[-1][i].set_zorder(2)
            else:
                hist2[-1][i].set_facecolor(c2_on_c1)
        return (hist1, hist2)

    else: 
        ax.legend(loc='upper left')
        return hist1
'''
def boxplot(ax, data, color='k', sym='b.'):
    """ Create a nice-looking boxplot with the data in data. Columns
    should be the different samples. sym handles the outlier mark,
    default is no mark. """

    data = np.asarray(data)
    # Shortcut method if there is no nan data
    if not np.any(np.isnan(data)): cdata = data
    else:
        cdata = [col[~np.isnan(col)] for col in data.T]

    bp = ax.boxplot(cdata, sym=sym, widths=0.65)
    plt.setp(bp['medians'], color=color, linewidth=0.75,
             solid_capstyle='butt')
    plt.setp(bp['boxes'], color=color, linewidth=0.5)
    plt.setp(bp['whiskers'], color=color, linewidth=0.5, linestyle='--',
             dashes=(4,3))
    plt.setp(bp['caps'], color=color, linewidth=0.5)
    plt.setp(bp['fliers'], markerfacecolor=color, markeredgecolor=color)
    hide_spines(ax)

def hide_spines(ax):
    """Hides the top and rightmost axis spines from view for all active
    figures and their respective axes."""
    # Disable spines.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Disable ticks.
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def density_contour(ax, xdata, ydata, nbins_x, nbins_y, range=None,
                    levels=None, **contour_kwargs):
    """ Create a density contour plot.

    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a
        new figure
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """

    H, xedges, yedges = np.histogram2d(xdata, ydata,
                                       bins=(nbins_x, nbins_y),
                                       normed=True, range=range)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))

    pdf = (H*(x_bin_sizes*y_bin_sizes))

    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T

    contour = ax.contour(X, Y, Z, levels=levels, colors='0.2')
    ax.contourf(X, Y, Z, levels=levels, cmap=matplotlib.cm.PuBu,
                **contour_kwargs)

    return contour

def lighten_color(color, degree):
    cin = matplotlib.colors.colorConverter.to_rgb(color)
    cw = np.array([1.0]*3)
    return tuple(cin + (cw - cin)*degree)

def color_range(NUM_COLORS, cm=None):
    if cm is None: cm = matplotlib.cm.get_cmap('gist_rainbow')
    return (cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS))

def jitter_uni(x_values, x_range = None):
    """Adds jitter in the x direction according to 
    http://matplotlib.1069221.n5.nabble.com/jitter-in-matplotlib-td12573.html
    where we use a uniform distribution in x."""
    if len(x_values)==1:
        print("No need to jitter_uni, single x value")
        return
        
    from scipy import stats
    
    if x_range == None:
        x_range = (np.max(x_values) - np.min(x_values))/len(np.unique(x_values))
    
    jitter = x_values + stats.uniform.rvs(-x_range,2*x_range,len(x_values))
    return jitter

def jitter_norm(y_values, y_range = None):
    """Adds jitter in the y direction according to 
    http://stackoverflow.com/questions/8671808/matplotlib-preventing-overlaying-datapoints
    where we use a normal distribution in y."""
    if len(y_values)==1:
        print("No need to jitter_norm, single y value")
        return
    
    y_range = .01*(np.max(y_values)-min(y_values))
    jitter = y_values + np.random.randn(len(y_values)) * y_range
    
    return jitter
