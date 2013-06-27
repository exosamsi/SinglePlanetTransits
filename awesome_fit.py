'''
Created on Jun 19, 2013

@author: Robert C. Morehead, rcmorehead@gmail.com 
Version 0.9.2

MIT License (MIT)

Copyright (c) 2013 SAMSI Kepler Data Working Group 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*******

Welcome to the awesome planet transit fitter! 

This is a quick and dirty non-physical transit fitting routine for 
preliminary transit searches where possible transits have already been identified
(i.e. from Planethunters.org) and clipped out of the light curve. 

This module requires numpy, matplotlib and mpfit.py

mpfit.py can be found at Adam Ginsburg's Google Code Page:
https://code.google.com/p/agpy/source/browse/trunk/mpfit/?r=399

Usage Example:
---------------
import awesome_fit

time = some data 
flux = some data
flux_err = some data

parameters, errors, fit_metrics = awesome_fit.awesome_fit(time,flux,flux_err, name='transit')  



Created at SAMSI Summer 2013 Program: 
Modern Statistical and Computational Methods for Analysis
of Kepler Data: June 10-28, 2013

Special thanks to T. Barclay, M. Payne and M. Schwamb  
'''


import numpy as np 
import mpfit
import pylab as plt


def box(X,Q_1,Q_2,f_0,d):
    '''
    Computes a box-shaped transit for some vector X, with width T, depth d, 
    mid-transit time x_0 and out of transit flux baseline f_0
    
    Returns a numpy array of floats of Y values 
    '''
    
    #Compute a box model 
    Y = []
    for x in X :
        y = f_0
        if x < Q_1 : 
            y = f_0
        if x >= Q_1 and x < Q_2 :
            y = f_0 - d
        if x >= Q_2 :
            y = f_0
        
        Y.append(y)
        
    return np.array(Y)

def dumb_detrendy(X,Y,npoints=5):
    '''
    The dumbest linear detrending possible. Take the first and last npoints in 
    x and y and median them together. Next draw a line between the median points. 
    Subtract that from y and return the new ys.
    
    Returns an numpy float array
    '''

    x1 = np.median(X[:npoints])
    x2 = np.median(X[-npoints:])
    y1 = np.median(Y[:npoints])
    y2 = np.median(Y[-npoints:])
    
    line = (y2-y1)/(x2-x1) * (X - x1) + y1
    
    return Y - line + y1,(y2-y1)/(x2-x1)


def trap(X,T,t,d,x_0,f_0):
    '''
    Computes a trapezoid-shaped transit for some vector X, with width T, inner-width t, depth d, 
    mid-transit time x_0 and out of transit flux baseline f_0
    
    Returns a numpy array of floats of Y values 
    '''
    
    Q_1 = x_0 - T
    Q_4 = x_0 + T
    Q_2 = x_0 - t
    Q_3 = x_0 + t
    
    if T == t : 
        m,m2 = 0,0
    else:
        m = ((f_0-d)-f_0)/(Q_2-Q_1)
        m2 = (f_0-(f_0-d))/(Q_4-Q_3)   
    
    Y = []
    y=f_0
    for x in X :    
        if x < Q_1 : 
            y = f_0
        if x >= Q_1 and x < Q_2 : 
            y = m*x - m*Q_1 + f_0 
        if x >= Q_2 and x < Q_3 : 
            y = f_0 - d 
        if x >= Q_3 and x < Q_4 : 
            y = m2*x - m2*Q_3 + (f_0- d) 
        if x >= Q_4 : 
            y = f_0
        
        Y.append(y)
        
        
    
    return np.array(Y)

        
def prayerbead(x,y,yerr,T,t,d,x_0,f_0,):  
    '''
    prayerbead(x,y,yerr,T,t,d,x_0,f_0,)
    
    Returns the 1-sigma errors on the parameters of a given fit
    from awesome_fit() by using the 'Prayer bead' method.
    
        Parameters
    -----------
    x: array_like
        x-coordinates of the data, typically time.
    y: array-like
        y-coordinates of the data, typically relative flux.
    yerr: array-like   
        the error on the y-coordinates.
    T:  float
        Transit duration of 'best fit'
    t: float
        Inner duration of 'best fit'
    d: float
        depth of 'best fit'
    x_0: float
        x-coordinate of the middle of trapezoid of 'best fit'
    f_0: float
        y baseline level of 'best fit'
    
    Returns
    --------
    out: List of floats
        1-sigma errors as parameters above. [T_err,t_err,d_err,x_0_err,f_0_err]

    
      
    '''
    #Make a dict to store fit values
    param_keys = ['T','t','d','x_o','f_0']
    params_store = {}
    for k in param_keys:
        params_store[k] = []
    
    #Compute the residuals from the best fit
    y_res = y-trap(x,T,t,d,x_0,f_0)
    
    #Begin sliding through the residuals
    for n in range(len(y_res)) :
        if n == 0 :
            pass
        else :
            y_res = np.roll(y_res, 1)
        
        #add residuals to y
        y2 = y + y_res 
        
        #refit using best fit as starting guess
        G_0 = [T,t,d,x_0,f_0]
        fa = {'x':x, 'y':y2, 'err':yerr}
        m = mpfit.mpfit(trapfit, G_0, functkw=fa,quiet=1) 
        
        #Store the results
        for j,key in enumerate(param_keys):
            params_store[key].append(m.params[j])

    #Find the standard deviation of all the fits
    out = []
    for j,key in enumerate(param_keys):           
        out.append(np.std(params_store[key]))
 
        
    return out


def trapfit(p, fjac=None, x=None, y=None, err=None):
    '''
    Trapezoid model set up for mpfit. See mpfit documentation.  
    '''
    model = trap(x, p[0], p[1], p[2], p[3], p[4])
    status = 0
    return [status, (y-model)/err]


def awesome_print(P,E,M,name='Transit'):
    '''
    Convenience function to print output of awesome_fit
    '''

    print '*** Fit for %s ***' % (name)
    keys = ['D','t','d','T_0','F_0']
    for k,key in enumerate(keys):
        print '%s=%s +/- %s' % (key,P[k],E[k])
        
    print 'Shape=%s' % M[0]
    print 'CHI^2=%s' % M[1]
    print 'Slope=%s' % M[2]
    
    return

def awesome_fit(x,y,yerr, name='transit', plot=True, show_plot=False, min_width=0,
                max_width=4, step=0.01, N_ML=5, verbose=False ):
    '''
    awesome_fit(x,y,yerr, name='transit', plot=True, show_plot=False, min_width=0,
                max_width=4, step=0.01, N_ML=5, verbose=False )
    
    Scan through a light curve with multiple boxes and return the 'best' one.
    Then fit a trapezoid with initial guesses from the box fit using multiple gaussian perturbed 
    Levenberg-Marquardt least-squares minimizations and then calculate the fit errors 
    of the resulting best fit by 'prayer bead' analysis.
    
    Parameters
    -----------
    x: array_like
        x-coordinates of the data, typically time.
    y: array-like
        y-coordinates of the data, typically relative flux.
    yerr: array-like   
        the error on the y-coordinates.
    name: string, optional, default 'transit' 
        Name used when plotting and saving the plot pdf. Should not contain spaces.
        Can also include a path to an existing directory and be parsed for '/'. 
        For example: name = '/scratch/transits/lightcurves/KOI-XXX'
        will create KOI-XXX.pdf in /scratch/transits/lightcurves/
        but the only the name KOI-XXX will be in the plot. 
    plot: Boolean True or False, optional, default True 
        Make a plot of the light curve fit and save it as a .pdf file. 
    show_plot: Boolean True or False, optional, default False 
        Show the plot on the screen. Useful if you are working with data interactively. 
    min_width, float, optional, default 0
        Minimum width in same units as x to start box scan with. Default assumes days.
    max_width, float, optional, default 4
        Minimum width in same units as x to start box scan with. Default assumes days.
    step, float, optional, default 0.01
        Step-size for scan box widths in same units as x to start box scan with.
        Default assumes days.
    N_ML: integer, optional, default 5
        Number of gaussian perturbed Levenberg-Marquardt fits to try to find the best fit. 
        In testing 5 was sufficient. 
    verbose: Boolean True or False, optional, default False
        Verbose output to stdout. Use this if you want to watch awesome_fit be awesome. 
         
        
    Returns
    --------
    p: List of floats
        This contains the fit parameters; 1st-4th contact transit duration, 
        2nd-3rd contact duration, transit depth, time of mid-transit, out of 
        transit flux level (after simple linear detrending)
    e: List of floats
        1-sigma errors as parameters above. Derived from 'prayer-bead' estimation 
    m: List of floats (2nd-3rd duration)/1st-4th
        Fit metrics; shape {(2nd-3rd duration)/(1st-4th duration)}, Reduced CHI^2
        of best fits, slope of simple linear detrending.  
    
    Example
    --------    
    import awesome_fit
    
    #Load your light curve (exercise left to the reader)
    time = some.data 
    flux = some.data
    flux_err = some.data

    #Fit the 'transit'
    p,e,m = awesome_fit.awesome_fit(time,flux,flux_err, name='my_transit')  
    
    #Print out the results in easy to read format 
    
    awesome_fit.awesome_print(P,E,M,name='my_transit')
    
    
        
    '''
    #Data must not contain nan.
    if  True in np.isnan(x) or True in np.isnan(y) or True in np.isnan(yerr) :
        print 'Inputs in %s contain NANs! Remove them and try again.' % name
        return np.nan,np.nan,np.nan
    
    if verbose == True : 
        shh = 1
    else :
        shh = 0 
    
    #If there is a linear slope, take it out of the data     
    y,slope = dumb_detrendy(x,y)
    
    #Inital f_0 guess is the median of the first and last five data points.
    f_0 = np.median(list(y[:5])+list(y[-5:]))    
    
    #Inital depth is the median of the 4 lowest data points.    
    d_0 = f_0-np.median(sorted(y)[:4])
    
    max_half = max_width/2
    min_half = min_width
    
    #Make an array of box midpoints to try fits at.
    x_0s = np.arange(x[0],x[-1],step)
    
    #Begin scanning boxes through the data.
    boxes = []
    for w in np.arange(min_half,max_half,step) :
        for j,x_0 in enumerate(x) :
            
            Q1 = x_0-w
            Q2 = x_0+w
            
            F = box(x,Q1,Q2,f_0,d_0)
            boxes.append((sum(((y-F)/yerr)**2)/(len(x)-5),x_0,d_0,Q1,Q2,f_0,Q2-Q1))

    #Find the lowest CHI^2
    boxes.sort(key=lambda tup: tup[0])
    box_keys = ['CHI^2','t_0','d_0','Q1','Q4','f_0','TD']
    
    #dict of initial box-based parameter guesses
    P = {}
    for j,k in enumerate(box_keys) :
        P[k] = boxes[0][j]
    
    if verbose == True :
        print 'Box fit:'
        for j in box_keys :
            print  '%s=%.4f' % (j,P[j])
        print ''
        
    #Storage for the ML fitting
    CHI2 = []
    params_struct = {}

    #Start gaussian perturbed ML fits. 
    for n in range(N_ML) :
        if verbose == True : print 'M-L #:%s' % n

        #Perturb T guess with a gaussian. 
        T = np.abs(np.random.normal(loc=P['TD'],scale=0.2*P['TD'])) 
        #Inner width guess is a uniformly random fraction the perturbed T guess
        Tt = T*np.random.uniform()
        
        fa = {'x':x, 'y':y, 'err':yerr}
        G_0 = [T,Tt,P['d_0'],P['t_0'],P['f_0']]
        
        if verbose == True : 
            print 'Initial guesses:'    
            for j in range(len(G_0)):    
                print 'P%s = %s' % (j,G_0[j])
        
        #Run the ML least-squares fit 
        m = mpfit.mpfit(trapfit, G_0, functkw=fa,quiet=shh) 
        
        if verbose == True : print m.status
        if verbose == True : print m.perror
        if verbose == True : print m.fnorm
        if (m.status <= 0): print 'error message = ', m.errmsg
        
        #Store the CHI^2s
        CHI2.append(m.fnorm)
        params_struct[str(m.fnorm)] = m
    
    #Fit the fit with the lowest CHI^2, that's the best one! 
    p = params_struct[str(min(CHI2))]
    

    
    p = p.params
    
    #Calculate some useful parameters 
    re_chi = min(CHI2)/(len(x)-6)
    shape = p[1]/p[0]
    
    if verbose == True : 
        print P['CHI^2'],min(CHI2)/(len(x)-6)
        for j in range(len(p)):    
            print 'P%s = %s' % (j,p[j])
  
    #Calculate fit errors using 'prayer-bead'  
    err = prayerbead(x,y,yerr,p[0], p[1], p[2], p[3], p[4])
   
    
    #Plots are useful, let's make one!
    if plot == True :
        
        #Set up plot
        f1 = plt.figure(figsize=(18, 6))
        plt.subplots_adjust(hspace=0.4,wspace=0.07)
        
        #Let's make some nice fit lines to plot
        model = trap(x, p[0], p[1], p[2], p[3], p[4]) 
        plot_step = (x[-1]-x[0])/float(3*len(x))
        fit_plot_x = np.arange(x[0],x[-1]+plot_step,plot_step)
        fit_plot_y = trap(fit_plot_x, p[0], p[1], p[2], p[3], p[4])
        
        
        #Light curve plot
        ax1 = plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=2)
        plt.title(name.split('/')[-1],fontsize=24)
    

        plt.errorbar(x,y,yerr=yerr,fmt='ko',elinewidth=3, capsize=0, alpha=.5)
        plt.plot(fit_plot_x,fit_plot_y,'m-',lw=3)
        plt.xlim(x[0],x[-1])
        plt.ylabel('Flux',fontsize=18)
        
        #Resdiuals plot
        ax2 = plt.subplot2grid((3,3), (2,0), colspan=2)
    
        plt.plot(x,y-model,'s',alpha=1)
        plt.axhline(0,ls='--',c='k',alpha=.5)
        plt.xlim(x[0],x[-1])
        plt.ylabel('Residuals',fontsize=18)
        plt.xlabel('Time',fontsize=18)
        
        #Panel with the fit parameters
        ax3 = plt.subplot2grid((3,3), (0,2), rowspan=3)
        ax3.axes.get_xaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)
        plt.title('Fit Parameters',fontsize=24)
        params_string = r'''
        $D$: {0:.2e} +- {1:.2e}
        
        $t$: {2:.2e} +- {3:.2e}
        
        $\Delta F$: {4:.2e} +- {5:.2e}
        
        $T_0$: {6:.3f} +- {7:.2e}
        
        $F_0$: {8:.2e} +- {9:.2e}
        
        
        $t/D$: {10:.3f}
        
        $\chi^2$: {11:.3f}
        
        $m$: {12:.2e}
        '''.format(p[0],err[0],p[1],err[1],p[2],err[2],p[3],err[3],p[4],err[4],shape,re_chi,slope)
        plt.text(0,0.05,s=params_string,fontsize=16)
        
        #All done, enjoy your pretty plot.
        f1.savefig('%s.pdf' % name)
        
        if verbose == True : print 'Plot saved as %s' % name
        
        if show_plot == True : plt.show()
    
    return p,err,[shape,re_chi,slope]

if __name__ == '__main__':
    print "awesome_fit is so awesome, you gotta import me, I don't do 'nuthing on my own!"
    