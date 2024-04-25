"""
Calculate the wavelet and its significance.
"""
from __future__ import division, absolute_import
import numpy as np
from scipy.special._ufuncs import gamma, gammainc
from scipy.optimize import fminbound as fmin
from scipy.fftpack import fft, ifft

__author__ = "Juhyeong Kang"
__email__ =  "jhkang@astro.snu.ac.kr"

__all__ = ['Wavelet', 'WaveCoherency']

class Wavelet:
    """
    Compute the wavelet transform of the given data
    with sampling rate dt.
    
    By default, the MORLET wavelet (k0=6) is used.
    The wavelet basis is normalized to have total energy=1
    at all scales.
            
    Parameters
    ----------
    data : `~numpy.ndarray`
        The time series N-D array.
    dt : `float`
        The time step between each y values.
        i.e. the sampling time.
    axis: `int`
        The axis number to apply wavelet, i.e. temporal axis.
            * Default is 0
    dj : `float` (optional)
        The spacing between discrete scales.
        The smaller, the better scale resolution.
            * Default is 0.25
    s0 : `float` (optional)
        The smallest scale of the wavelet.  
            * Default is 2 * dt.
    j : `int` (optional)
        The number of scales minus one.
        Scales range from s0 up to s_0 * 2^{j dj}, to give
        a total of j+1 scales.
            * Default is j=log_2(n dt/(s_0 dj)).
    mother : `str` (optional)
        The mother wavelet function.
        The choices are 'MORLET', 'PAUL', or 'DOG'
            * Default is **'MORLET'**
    param  : `int` (optional)
        The mother wavelet parameter.
        For **'MORLET'** param is k0, default is **6**.
        For **'PAUL'** param is m, default is **4**.
        For **'DOG'** param is m, default is **2**.
    pad : `bool` (optional)
        If set True, pad time series with enough zeros to get
        N up to the next higher power of 2.
        This prevents wraparound from the end of the time series
        to the beginning, and also speeds up the FFT's 
        used to do the wavelet transform.
        This will not eliminate all edge effects.
    
    Notes
    -----
        This function based on the IDL code WAVELET.PRO written by C. Torrence, 
        and Python code waveletFuncitions.py written by E. Predybayalo.
    
    References
    ----------
    Torrence, C. and Compo, G. P., 1998, A Practical Guide to Wavelet Analysis, 
    *Bull. Amer. Meteor. Soc.*, `79, 61-78 <http://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf>`_.
    http://paos.colorado.edu/research/wavelets/
    
    Example
    -------
    >>> from fisspy.analysis import wavelet
    >>> res = wavelet.wavelet(data,dt,dj=dj,j=j,mother=mother,pad=True)
    >>> wavelet = res.wavelet
    >>> period = res.period
    >>> scale = res.scale
    >>> coi = res.coi
    >>> power = res.power
    >>> gws = res.gws
    >>> res.plot()
    """
    
    def __init__(self, data, dt, axis=0, dj=0.1, s0=None, j=None,
                 mother='MORLET', param=False, pad=True):

        
        shape0 = np.array(data.shape)
        self.n0 = shape0[axis]
        shape = np.delete(shape0, axis)
        self.axis = axis
        
        if not s0:
            S0 = 2*dt
        if not j:
            j = int(np.log2(self.n0*dt/S0)/dj)
        else:
            j=int(j)
        
        self.s0 = S0
        self.j = j
        self.dt = dt
        self.dj = dj
        self.mother = mother.upper()
        self.param = param
        self.pad = pad
        self.axis = axis
        self.data = data
        self.ndim = data.ndim
        
        #padding
        if pad:
#            power = int(np.log2(self.n0)+0.4999)
            power = int(np.log2(self.n0))
            self.npad = 2**(power+1)-self.n0
            self.n = self.n0 + self.npad
        else:
            self.n = self.n0
        
        #wavenumber
        k1 = np.arange(1,self.n//2+1)*2.*np.pi/self.n/dt
        k2 = -k1[:int((self.n-1)/2)][::-1]
        k = np.concatenate(([0.],k1,k2))
        
        #Scale array
        self.scale = self.s0*2**(np.arange(self.j+1,dtype=float)*dj)
        
        #base return
        self._motherFunc(k)
        self.coi *= self.dt*np.append(np.arange((self.n0+1)//2),
                                      np.arange(self.n0//2-1,-1,-1))
        
        #array handeling
        order_ini = np.arange(data.ndim)
        o1 = np.delete(order_ini, axis)
        o2 = np.concatenate([o1, [axis]])
        tdata = data.transpose(o2)
        indata = tdata.reshape([shape.prod(), self.n0])
        
        wshape = np.concatenate([shape, [self.j+1, self.n0]])
        self.wavelet = np.empty(np.concatenate([[shape.prod()], [self.j+1, self.n0]]),
                             dtype=complex)
        for i, y in enumerate(indata):
            self.wavelet[i] = self._getWavelet(y)[:,:self.n0]
#        self.wavelet = self._getWavelet(indata)[:,:,:self.n0]
        
        
        self.wavelet = self.wavelet.reshape(wshape)
        self.power = np.abs(self.wavelet)**2
        self.gws = self.power.mean(axis=-1)
        
    def _getWavelet(self, y):

#        x = y - y.mean(axis=-1)[:,None]
        x = y - y.mean(axis=-1)
        
        #reconstruct the time series to analyze if set pad
        if self.pad:
#            shape = y.shape
#            self.padding = np.zeros([shape[0], self.npad])
            self.padding = np.zeros(self.npad)
            x = np.concatenate((x, self.padding), axis=-1)
        
        # FFT
        fx = fft(x)
        
#        res = ifft(fx[:,None,:]*self.nowf)
        res = ifft(fx*self.nowf)
        return res
        
    
    def iwavelet(self, wavelet, scale):
        #%% should be revised (period range option)
        """
        Inverse the wavelet to get the time-series
        
        Parameters
        ----------
        wavelet : ~numpy.ndarray
            wavelet.
        
        Returns
        -------
        iwave : ~numpy.ndarray
            Inverse wavelet.
        
        Notes
        -----
            This function based on the IDL code WAVELET.PRO written by C. Torrence, 
            and Python code waveletFuncitions.py written by E. Predybayalo.
        
        References
        ----------
        Torrence, C. and Compo, G. P., 1998, A Practical Guide to Wavelet Analysis, 
        *Bull. Amer. Meteor. Soc.*, `79, 61-78 <http://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf>`_.\n
        http://paos.colorado.edu/research/wavelets/
            
        Example
        -------
        >>> iwave = res.iwavelet(wavelet)
        """
        scale2=1/scale**0.5
        
        self._motherParam()
        if self.cdelta == -1:
            raise ValueError('Cdelta undefined, cannot inverse with this wavelet')
        
        if self.mother == 'MORLET':
            psi0=np.pi**(-0.25)
        elif self.mother == 'PAUL':
            psi0=2**self.param*gamma(self.param+1)/(np.pi*gamma(2*self.param+1))**0.5
        elif self.mother == 'DOG':
            if not self.param:
                self.param=2
            if self.param==2:
                psi0=0.867325
            elif self.param==6:
                psi0=0.88406
        
        iwave=self.dj*self.dt**0.5/(self.cdelta*psi0)*np.dot(scale2, wavelet.real)
        return iwave
    
    def _motherFunc(self, k):
        """
        Compute the Fourier factor and period.
        
        Parameters
        ----------
        mother : str
            A string, Equal to 'MORLET' or 'PAUL' or 'DOG'.
        k : 1d ndarray
            The Fourier frequencies at which to calculate the wavelet.
        scale : ~numpy.ndarray
            The wavelet scale.
        param : int
            The nondimensional parameter for the wavelet function.
        
        Returns
        -------
        nowf : ~numpy.ndarray
            The nonorthogonal wavelet function.
        period : ~numpy.ndarrary
            The vecotr of "Fourier" periods (in time units)
        fourier_factor : float
            the ratio of Fourier period to scale.
        coi : int
            The cone-of-influence size at the scale.
        
        Notes
        -----
            This function based on the IDL code WAVELET.PRO written by C. Torrence, 
            and Python code waveletFuncitions.py written by E. Predybayalo.
        
        References
        ----------
        Torrence, C. and Compo, G. P., 1998, A Practical Guide to Wavelet Analysis, 
        *Bull. Amer. Meteor. Soc.*, `79, 61-78 <http://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf>`_.\n
        http://paos.colorado.edu/research/wavelets/
            
        """
        kp = k > 0.
        scale2 = self.scale[:, None]
        pi = np.pi
        
        if self.mother == 'MORLET':
            if not self.param:
                self.param = 6.
            expn = -(scale2*k-self.param)**2/2.*kp
            norm = pi**(-0.25)*(self.n*k[1]*scale2)**0.5
            self.nowf = norm*np.exp(expn)*kp*(expn > -100.)
            self.fourier_factor = 4*pi/(self.param+(2+self.param**2)**0.5)
            self.coi = self.fourier_factor/2**0.5
            
        elif self.mother == 'PAUL':
            if not self.param:
                self.param = 4.
            expn = -scale2*k*kp
            norm = 2**self.param*(scale2*k[1]*self.n/(self.param*gamma(2*self.param)))**0.5
            self.nowf = norm*np.exp(expn)*((scale2*k)**self.param)*kp*(expn > -100.)
            self.fourier_factor = 4*pi/(2*self.param+1)
            self.coi = self.fourier_factor*2**0.5
            
        elif self.mother == 'DOG':
            if not self.param:
                self.param = 2.
            expn = -(scale2*k)**2/2.
            norm = (scale2*k[1]*self.n/gamma(self.param+0.5))**0.5
            self.nowf = -norm*1j**self.param*(scale2*k)**self.param*np.exp(expn)
            self.fourier_factor = 2*pi*(2./(2*self.param+1))**0.5
            self.coi = self.fourier_factor/2**0.5
        else:
            raise ValueError('Mother must be one of MORLET, PAUL, DOG\n'
                             'mother = %s' %repr(self.mother))
        self.period = self.scale*self.fourier_factor
        self.freq = self.dt/self.period
    
    def _motherParam(self):
        """
        Get the some values for given mother function of wavelet.
        
        Parameters
        ----------
        mother : str
        param : int
            The nondimensional parameter for the wavelet function.
            
        Returns
        -------
        fourier_factor : float
            the ratio of Fourier period to scale.
        dofmin : float
            Degrees of freedom for each point in the wavelet power.
            (either 2 for MORLET and PAUL, or 1 for the DOG)
        cdelta : float
            Reconstruction factor.
        gamma_fac : float
            decorrelation factor for time averaging.
        dj0 : float
            factor for scale averaging.
        
        Notes
        -----
            This function based on the IDL code WAVELET.PRO written by C. Torrence, 
            and Python code waveletFuncitions.py written by E. Predybayalo.
        
        References
        ----------
        Torrence, C. and Compo, G. P., 1998, A Practical Guide to Wavelet Analysis, 
        *Bull. Amer. Meteor. Soc.*, `79, 61-78 <http://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf>`_.\n
        http://paos.colorado.edu/research/wavelets/
        
            
        """
        self.cdelta = -1
        self.gamma_fac = -1
        self.dj0 = -1
        if self.mother == 'MORLET':
            self.dofmin=2.
            if self.param == 6.:
                self.cdelta = 0.776
                self.gamma_fac = 2.32
                self.dj0 = 0.60
            elif self.param == 12:
                self.cdelta = 0.38
                self.dj0 = 0.60
            elif self.param == 18:
                self.cdelta = 0.27
                self.dj0 = 0.60
        elif self.mother == 'PAUL':
            if not self.param:
                self.param = 4.
            self.dofmin = 2.
            if self.param == 4.:
                self.cdelta = 1.132
                self.gamma_fac = 1.17
                self.dj0 = 1.5
            else:
                self.cdelta = -1
                self.gamma_fac = -1
                self.dj0 = -1
        elif self.mother == 'DOG':
            if not self.param:
                self.param = 2.
            self.dofmin = 1.
            if self.param == 2.:
                self.cdelta = 3.541
                self.gamma_fac = 1.43
                self.dj0 = 1.4
            elif self.param ==6.:
                self.cdelta = 1.966
                self.gamma_fac = 1.37
                self.dj0 = 0.97
            else:
                self.cdelta = -1
                self.gamma_fac = -1
                self.dj0 = -1
        else:
            raise ValueError('Mother must be one of MORLET, PAUL, DOG')
        
        
    def saveWavelet(self, savename):
        """
        Save the wavelet spectrum as .npz file.
        
        Parameters
        ----------
        savename: `str`
            filename to save the wavelet data.
        
        """
        
        np.savez(savename, wavelet=self.wavelet,
                 period=self.period, scale=self.scale,
                 coi=self.coi, dt=self.dt, dj=self.dj, axis=self.axis,
                 s0=self.s0, j=self.j, mother=self.mother,
                 param=self.param)
        
    def waveSignif(self, y, sigtest=0, lag1=0., siglvl=0.95, dof=-1,
                    gws=False, confidence=False):
        """
        Compute the significance levels for a wavelet transform.
        
        Parameters
        ----------
        y : float or ~numpy.ndarray
            The time series, or the variance of the time series.
            If this is a single number, it is assumed to be the variance.
        sigtest : (optional) int
            Allowable values are 0, 1, or 2
            if 0 (default), then just do a regular chi-square test
                i.e. Eqn (18) from Torrence & Compo.
            If 1, then do a "time-average" test, i.e. Eqn (23).
                in this case, dof should be set to False,
                the nuber of local wavelet spectra 
                that were averaged together.
                For the Global Wavelet Spectrum(GWS), this would be N,
                where N is the number of points in y
            If 2, then do a "scale-average" test, i.e. Eqns (25)-(28).
                In this case, dof should be set to a two-element vector,
                which gives the scale range that was averaged together.
                e.g. if one scale-averaged scales between 2 and 8,
                then dof=[2,8]
        lag1 : (optional) float
            LAG 1 Autocorrelation, used for signif levels.
                * Default is 0.
        siglvl : (optional) float
            Significance level to use.
                * Default is 0.95
        dof : (optional) float
            degrees-of-freedom for sgnif test.
                * Default is -1, and it means the False.
    
                
        Returns
        -------
        signif : ~numpy.ndarray
                Significance levels as a function of scale.
            
        Notes
        -----
        IF SIGTEST=1, then DOF can be a vector (same length as SCALEs), 
        in which case NA is assumed to vary with SCALE. 
        This allows one to average different numbers of times 
        together at different scales, or to take into account 
        things like the Cone of Influence.\n
        See discussion following Eqn (23) in Torrence & Compo.\n
        This function based on the IDL code WAVE_SIGNIF.PRO written by C. Torrence, 
        and Python code waveletFuncitions.py written by E. Predybayalo.
        
        References
        ----------
        Torrence, C. and Compo, G. P., 1998, A Practical Guide to Wavelet Analysis, 
        *Bull. Amer. Meteor. Soc.*, `79, 61-78 <http://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf>`_.\n
        http://paos.colorado.edu/research/wavelets/
        
        Example
        -------
        >>> signif=wavelet.wave_signif(y,dt,scale,2,mother='morlet',dof=[s1,s2],gws=gws)
        
        """
        if len(np.atleast_1d(y)) == 1:
            var = y
        else:
            var = np.var(y)
        
        j = len(self.scale)
        
        self._motherParam()
        try:
            len(gws)
            fft_theor = gws.copy()
        except:
            fft_theor = (1-lag1**2)/(1-2*lag1*np.cos(self.freq*2*np.pi)+lag1**2)
            fft_theor*=var
    
    
        signif = fft_theor.copy()
        
        if sigtest == 0:
            dof = self.dofmin
            signif = fft_theor * self._chisquareInv(siglvl, dof)/dof
            if confidence:
                sig = (1.-siglvl)/2.
                chisqr = dof/np.array((self._chisquareInv(1-sig, dof),
                                       self._chisquareInv(sig, dof)))
                signif = np.dot(chisqr[:,None], fft_theor[None,:])
        elif sigtest == 1:
            if self.gamma_fac == -1:
                raise ValueError('gamma_fac(decorrelation facotr) not defined for '
                                 'mother = %s with param = %s'
                                 %(repr(self.mother), repr(self.param)))
            if len(np.atleast_1d(dof)) != 1:
                pass
            elif dof == -1:
                dof = np.zeros(j)+self.dofmin
            else:
                dof = np.zeros(j)+dof
            dof[dof <= 1] = 1
            dof = self.dofmin * (1 + (dof * self.dt/self.gamma_fac/self.scale)**2)**0.5
            dof[dof <= self.dofmin] = self.dofmin
            if not confidence:
                for i in range(j):
                    chisqr = self._chisquareInv(siglvl, dof[i]) / dof[i]
                    signif[i] = chisqr * fft_theor[i]
            else:
                signif = np.empty(2,j)
                sig = (1-siglvl)/2.
                for i in range(j):
                    chisqr = dof[i]/np.array((self._chisquareInv(1-sig,dof[i]),
                                            self._chisquareInv(sig, dof[i])))
                    signif[:,i] = fft_theor[i]*chisqr
        elif sigtest == 2:
            if len(dof) != 2:
                raise ValueError('DOF must be set to [s1,s2], the range of scale-averages')
            if self.cdelta == -1:
                raise ValueError('cdelta & dj0 not defined for'
                                 'mother = %s with param = %s' %(repr(self.mother), repr(self.param)))
            dj= np.log2(self.scale[1] / self.scale[0])
            s1 = dof[0]
            s2 = dof[1]
            avg = (self.period >= s1)*(self.period <= s2)
            navg = avg.sum()
            if not navg:
                raise ValueError('No valid scales between %s and %s' %(repr(s1), repr(s2)))
            s1 = self.scale[avg].min()
            s2 = self.scale[avg].max()
            savg = 1./(1./self.scale[avg]).sum()
            smid = np.exp(0.5*np.log(s1*s2))
            dof = (self.dofmin*navg*savg/smid)*(1+(navg*dj/self.dj0)**2)**0.5
            fft_theor = savg*(fft_theor[avg]/self.scale[avg]).sum()
            chisqr = self._chisquareInv(siglvl,dof)/dof
            if confidence:
                sig = (1-siglvl)/2.
                chisqr = dof/np.array((self._chisquareInv(1-sig, dof),
                                       self._chisquareInv(sig, dof)))
            signif = (self.dj*self.dt/self.cdelta/savg)*fft_theor*chisqr
        else:
            raise ValueError('Sigtest must be 0,1, or 2')
        self.signif = signif
        return signif
    
    def _chisquareInv(self,p,v):
        """
        Inverse of chi-square cumulative distribution function(CDF).
        
        Parameters
        ----------
        p : float
            probability
        v : float
            degrees of freedom of the chi-square distribution
        
        Returns
        -------
        x : float
            the inverse of chi-square cdf
            
        Example
        -------
        >>> result = chisquare_inv(p,v)
        
        """
        if not 0<p<1:
            raise ValueError('p must be 0<p<1')
        minv = 0.01
        maxv = 1
        x = 1
        tolerance = 1e-4
        while x+tolerance >= maxv:
            maxv*=10.
            x = fmin(self._chisquareSolve, minv, maxv, args=(p,v), xtol=tolerance)
            minv = maxv
        x*=v
        return x
        
    def _chisquareSolve(self,xguess,p,v):
        """
        Chisqure_solve
        
        Return the difference between calculated percentile and P.
        
        Written January 1998 by C. Torrence
        """
        pguess = gammainc(v/2,v*xguess/2)
        pdiff = np.abs(pguess - p)
        if pguess >= 1-1e-4:
            pdiff = xguess
        return pdiff
    
    def plot(self, lag1=0., levels=None, time=None, title=[None,None,None,None], figsize=(9,8)):
        """
        Plot Time Series, Wavelet Power Spectrum, 
        Global Power Spectrum and Scale-average Time Series.
        
        Parameters
        ---------
        lag1: (optional) `float`
            LAG 1 Autocorrelation, used for signif levels.
                * Default is 0.
        levels: list
            Contour levels to plot the wavelet spectrum.
        time: `~numpy.ndarray`
            time array.
        title: list
            title of the each figure.
        figsize: tuple
            figure size
        
        Example
        -------
        >>> ww = Wavelet(data, 0.25, dj=0.1, s0=0.25, j=9/0.1)
        >>> ww.plot()
        """
        
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib import ticker
        
        n = len(self.data)
        gs = GridSpec(7, 4)
        self.fig = plt.figure(figsize=figsize)
        self.axData = self.fig.add_subplot(gs[0:2, :3])
        self.axWavelet = self.fig.add_subplot(gs[2:5, :3], sharex=self.axData)
        self.axGlobal = self.fig.add_subplot(gs[2:5, 3], sharey=self.axWavelet)
        self.axScaleAvg = self.fig.add_subplot(gs[5:7, :3], sharex=self.axData)
        periodMax = self.period.max()
        periodMax = periodMax if periodMax<64 else 64
        
        if time is None:
            ttime = self.dt*np.arange(n)
        if levels is None:
            Levels = [0.05, 0.12,0.229,
                      0.45]
            
        # Plot Time Series
        if title[0] is None:
            self.axData.set_title('a) Time Series')
        else:
            self.axData.set_title(title)
        self.axData.set_ylabel('Value')
        self.axData.minorticks_on()
        self.axData.tick_params(which='both', direction='in')
        self.pData = self.axData.plot(ttime, self.data,
                                      color='k', lw=1.5)[0]
        self.axData.set_xlim(ttime[0], ttime[-1])
        
        # Contour Plot Wavelet Power Spectrum
        if title[1] is None:
            self.axWavelet.set_title('b) Wavelet Power Spectrum')
        else:
            self.axWavelet.set_title(title)
        self.axWavelet.set_ylabel('Period')
        self.axWavelet.minorticks_on()
        
        self.axWavelet.tick_params(which='both', direction='in')
        self.axWavelet.set_yscale('symlog', basey=2)
        self.axWavelet.yaxis.set_major_formatter(ticker.ScalarFormatter())
        self.axWavelet.ticklabel_format(axis='y',style='plain')
        self.axWavelet.set_ylim(periodMax, 0.5)
        
        wpower = self.power/self.power.max()
        self.contour = self.axWavelet.contourf(ttime, self.period,
                                               wpower, len(Levels),
                                               colors=['w'])
        self.contourIm = self.axWavelet.contourf(self.contour,
                                                 levels=Levels,
                                                 cmap=plt.cm.Spectral_r, extend='max')
        signif = self.waveSignif(self.data, sigtest=0, lag1=lag1, siglvl=0.90,
                                 gws=self.gws)
        sig90 = signif[:,None]
        sig90 = self.power/sig90
        
        self.axWavelet.contour(ttime, self.period, sig90, [-99,1] ,colors='r')
        self.axWavelet.fill_between(ttime, self.coi,
                                    self.period.max(), color='grey',
                                    alpha=0.4, hatch='x')
        
        # Plot Global Wavelet Spectrum
        if title[2] is None:
            self.axGlobal.set_title('c) Global')
        else:
            self.axGlobal.set_title(title)
        self.axGlobal.set_xlabel('Power')
        self.axGlobal.set_ylabel('')
        self.axGlobal.set_yscale('symlog', basey=2)
        self.axGlobal.minorticks_on()
        self.axGlobal.yaxis.set_major_formatter(ticker.ScalarFormatter())
        self.axGlobal.ticklabel_format(axis='y',style='plain')
        self.axGlobal.tick_params(which='both', direction='in')
        self.pGlobal = self.axGlobal.plot(self.gws, self.period,
                                          color='k', lw=1.5)[0]
        
        
        dof = n - self.scale
#        lag1 = 0.72
        gsig = self.waveSignif(self.data, sigtest=1, lag1=0,
                               dof=dof)
        self.pSig = self.axGlobal.plot(gsig,
                                       self.period, 
                                       'r--',
                                       lw=1.5)
        
        
        # Plot Scale-average Time Series
        if title[3] is None:
            self.axScaleAvg.set_title('d) Scale-average Time Series')
        else:
            self.axScaleAvg.set_title(title)
        self.axScaleAvg.set_xlabel('Time')
        self.axScaleAvg.set_ylabel('Avg')
        self.axScaleAvg.minorticks_on()
        self.axScaleAvg.tick_params(which='both', direction='in')
        
        period_mask = (self.period >= 2)*(self.period < 8)
        power_norm = self.power/self.scale[:,None]
        power_avg = self.dj*self.dt/self.cdelta*power_norm[period_mask,:].sum(0)
        self.pScaleAvg = self.axScaleAvg.plot(ttime,
                                              power_avg,
                                              color='k',
                                              lw=1.5)
        
        self.fig.tight_layout()







class WaveCoherency:
    def __init__(self, wave1, time1, scale1, wave2, time2, scale2,
                 dt=False, dj=False, coi=False, nosmooth=False):
        """
        Compute the wavelet coherency between two time series.
        
        Parameters
        ----------
        wave1 : ~numpy.ndarray
            Wavelet power spectrum for time series 1.
        time1 : ~numpy.ndarray
            A vector of times for time series 1.
        scale1 : ~numpy.ndarray
            A vector of scales for time series 1.
        wave2 : ~numpy.ndarray
            Wavelet power spectrum for time series 2.
        time2 : ~numpy.ndarray
            A vector of times for time series 2.
        scale2 : ~numpy.ndarray
            A vector of scales for time series 2.
        dt : (optional) float
            Amount of time between each Y value, i.e. the sampling time.
                If not input, then calculated from time1[1]-time1[0]
        dj : (optional) float
            The spacing between discrete scales.
                If not input, then calculated from scale1
        coi : (optional) ~numpy.ndarray
            The array of the cone-of influence.
        nosmooth : (optional) bool
            If True, then just compute the global_coher, global_phase, and
            the unsmoothed cross_wavelet and return.
        
        Returns
        -------
            
        cross_wavelet : ~numpy.ndarray
            The cross wavelet between the time series.
        time : ~numpy.ndarray
            The time array given by the overlap of time1 and time2.
        scale : ~numpy.ndarray
            The scale array of scale indices, given by the overlap of 
            scale1 and scale2.
        wave_phase : ~numpy.ndarray
            The phase difference between time series 1 and time series 2.
        wave_coher : ~numpy.ndarray
            The wavelet coherency, as a function of time and scale.
        global_phase : ~numpy.ndarray
            The global (or mean) phase averaged over all times.
        global_coher : ~numpy.ndarray
            The global (or mean) coherence averaged over all times.
        power1 : ~numpy.ndarray
            The wavelet power spectrum should be the same as wave1
            if time1 and time2 are identical, otherwise it is only the
            overlapping portion. If nosmooth is set,
            then this is unsmoothed, otherwise it is smoothed.
        power2 : ~numpy.ndarray
            same as power 1 but for time series 2.
        coi : ~numpy.ndarray
            The array of the cone-of influence.
            
        Notes
        -----
            This function based on the IDL code WAVE_COHERENCY.PRO written by C. Torrence, 
        
        References
        ----------
        Torrence, C. and Compo, G. P., 1998, A Practical Guide to Wavelet Analysis, 
        *Bull. Amer. Meteor. Soc.*, `79, 61-78 <http://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf>`_.\n
        http://paos.colorado.edu/research/wavelets/
        
        Example
        -------
        >>> res = wavelet.WaveCoherency(wave1,time1,scale1,wave2,time2,scale2,\
                                       dt,dj,coi=coi)
        >>> cross_wave = res.cross_wavelet
        >>> phase = res.wave_phase
        >>> coher = res.wave_coher
        >>> gCoher = res.global_coher
        >>> gCross = res.global_cross
        >>> gPhase = res.global_phase
        >>> power1 = res.power1
        >>> power2 = res.power2
        >>> time_out = res.time
        >>> scale_out = res.scale
        """
        if not dt: DT = time1[1]-time1[0]
        if not dj: DJ = np.log2(scale1[1]/scale1[0])
        if time1 is time2:
            t1s = 0
            t1e = len(time1)
            t2s = t1s
            t2e = t1e
        else:
            otime_start = min([time1.min(),time2.min()])
            otime_end = max([time1.max(),time2.max()])
            t1 = np.where((time1 >= otime_start)*(time1 <= otime_end))[0]
            t1s = t1[0]
            t1e = t1[-1]+1
            t2 = np.where((time2 >= otime_start)*(time2 <= otime_end))[0]
            t2s = t2[0]
            t2e = t2[-1]+1
        
        oscale_start = min([scale1.min(),scale2.min()])
        oscale_end = max([scale1.max(),scale2.max()])
        s1 = np.where((scale1 >= oscale_start)*(scale1 <= oscale_end))[0]
        s2 = np.where((scale2 >= oscale_start)*(scale2 <= oscale_end))[0]
        s1s = s1[0]
        s1e = s1[-1]+1
        s2s = s2[0]
        s2e = s2[-1]+1
        
        self.cross_wavelet = wave1[s1s:s1e,t1s:t1e]*wave2[s2s:s2e,t2s:t2e].conj()
        self.power1 = np.abs(wave1[s1s:s1e,t1s:t1e])**2
        self.power2 = np.abs(wave2[s2s:s2e,t2s:t2e])**2
        
        self.time = time1[t1s:t1e]
        self.scale = scale1[s1s:s1e]
        nj = s1e-s1s
        
        global1 = self.power1.sum(1)
        global2 = self.power2.sum(1)
        self.global_cross = self.cross_wavelet.sum(1)
        self.global_coher = np.abs(self.global_cross)**2/(global1*global2)
        self.global_phase = np.arctan(self.global_cross.imag/self.global_cross.real)*180./np.pi
        
        if not nosmooth:
            nt = (4*self.scale/DT)//2*4+1
            nt2 = nt[:,None]
            ntmax = nt.max()
            g = np.arange(ntmax) * np.ones((nj,1))
            wh = g >= nt2
            time_wavelet = (g-nt2//2)*DT/self.scale[:,None]
            wave_func = np.exp(-time_wavelet**2/2)
            wave_func[wh] = 0
            wave_func = (wave_func/wave_func.sum(1)[:,None]).real
            self.cross_wavelet = _fastConv(self.cross_wavelet, wave_func, nt2)
            self.power1 = _fastConv(self.power1, wave_func, nt2)
            self.power2 = _fastConv(self.power2, wave_func, nt2)
            scales = self.scale[:, None]
            self.cross_wavelet /= scales
            self.power1 /= scales
            self.power2 /= scales
            
            nw = int(0.6/DJ/2 + 0.5)*2-1
            weight = np.ones(nw)/nw
            self.cross_wavelet = _fastConv2(self.cross_wavelet, weight)
            self.power1 = _fastConv2(self.power1,weight)
            self.power2 = _fastConv2(self.power2,weight)
            
            self.wave_phase = 180./np.pi*np.arctan(self.cross_wavelet.imag/self.cross_wavelet.real)
            power3=self.power1*self.power2
            whp=power3 < 1e-9
            power3[whp]=1e-9
            self.wave_coher = (np.abs(self.cross_wavelet)**2/power3).real
    


def _fastConv(f, g, nt):
    """
    Fast convolution two given function f and g (method 1)
    """
    nf=f.shape
    ng=g.shape
    npad=2**(int(np.log2(max([nf[1],ng[1]])))+1)
    wh1=np.arange(nf[0],dtype=int)
    wh2=np.arange(nf[1],dtype=int)*np.ones((nf[0],1),dtype=int)-(nt.astype(int)-1)//2-1
    pf=np.zeros([nf[0],npad],dtype=complex)
    pg=np.zeros([nf[0],npad],dtype=complex)
    pf[:,:nf[1]]=f
    pg[:,:ng[1]]=g
    conv=ifft(fft(pf)*fft(pg[:,::-1]))
    result=conv[wh1,wh2.T].T
    return result

def _fastConv2(f, g):
    """
    Fast convolution two given function f and g (method2)
    """
    nf=f.shape
    ng=len(g)
    npad=2**(int(np.log2(max([nf[0],ng])))+1)
    
    wh1=np.arange(nf[1],dtype=int)
    wh2=np.arange(nf[0],dtype=int)*np.ones((nf[1],1),dtype=int)+ng//2
    pf=np.zeros([npad,nf[1]],dtype=complex)
    pg=np.zeros([npad,nf[1]],dtype=complex)
    pf[:nf[0],:]=f
    pg[:ng,:]=g[:,np.newaxis]
    conv=ifft(fft(pf,axis=0)*fft(pg,axis=0),axis=0)
    result=conv[wh2.T,wh1]
    return result