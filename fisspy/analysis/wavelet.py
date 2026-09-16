"""
Calculate the wavelet and its significance.
"""
from __future__ import division, absolute_import
import numpy as np
from scipy.special._ufuncs import gamma, gammainc
from scipy.optimize import fminbound as fmin
from scipy.fftpack import fft, ifft
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from scipy.signal import lfilter

__author__ = "Juhyung Kang"

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
        and Python code waveletFuncitions.py written by E. Predybayalo..
    
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
                 mother='MORLET', param=False, pad=False):

        
        shape0 = np.array(data.shape)
        self.n0 = shape0[axis]
        shape = np.delete(shape0, axis)
        self.axis = axis
        
        if not s0:
            S0 = 2*dt
        else:
            S0 = s0
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
        
    def waveSignif(self, y, sigtest=0, lag1=0., siglvl=0.95, dof=-1, gws=False, confidence=False):
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
            signif = fft_theor * _chisquareInv(siglvl, dof)/dof
            if confidence:
                sig = (1.-siglvl)/2.
                chisqr = dof/np.array((_chisquareInv(1-sig, dof),
                                       _chisquareInv(sig, dof)))
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
                    chisqr = _chisquareInv(siglvl, dof[i]) / dof[i]
                    signif[i] = chisqr * fft_theor[i]
            else:
                signif = np.empty(2,j)
                sig = (1-siglvl)/2.
                for i in range(j):
                    chisqr = dof[i]/np.array((_chisquareInv(1-sig,dof[i]),
                                            _chisquareInv(sig, dof[i])))
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
            chisqr = _chisquareInv(siglvl,dof)/dof
            if confidence:
                sig = (1-siglvl)/2.
                chisqr = dof/np.array((_chisquareInv(1-sig, dof),
                                       _chisquareInv(sig, dof)))
            signif = (self.dj*self.dt/self.cdelta/savg)*fft_theor*chisqr
        else:
            raise ValueError('Sigtest must be 0,1, or 2')
        self.signif = signif
        return signif
    

    
    def plot(self, lag1=0.72, siglvl=0.95, time=None, title=[None,None,None,None], figsize=(9,8)):
        """
        Plot Time Series, Wavelet Power Spectrum, 
        Global Power Spectrum and Scale-average Time Series.
        
        Parameters
        ---------
        lag1: (optional) `float`
            LAG 1 Autocorrelation, used for signif levels.
                * Default is 0.
        siglvl: (optional) `float`
            Significance level.
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
        
        
        
        n = len(self.data)
        periodMax = self.period.max()
        periodMax = periodMax if periodMax<64 else 64
        if time is None:
            ttime = self.dt*np.arange(n)
        else:
            ttime = time

        # Set figure
        gs = GridSpec(7, 4)
        self.fig = plt.figure(figsize=figsize)
        self.axData = self.fig.add_subplot(gs[0:2, :3])
        self.axWavelet = self.fig.add_subplot(gs[2:5, :3], sharex=self.axData)
        self.axGlobal = self.fig.add_subplot(gs[2:5, 3], sharey=self.axWavelet)
        self.axScaleAvg = self.fig.add_subplot(gs[5:7, :3], sharex=self.axData)
        
            
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
        
        # Wavelet Power Spectrum
        wpower = self.power/self.power.max()
        signif = self.waveSignif(self.data, sigtest=0, lag1=lag1, siglvl=siglvl)
        sig = self.power/signif[:,None]
        coi = self.coi.copy()
        coi[0] = 1e-3
        coi[-1] = 1e-3
        coi = np.log2(coi)
        lp = np.log2(self.period)
        dp = np.diff(lp).mean()
        t2 = ttime.copy()
        t2[0] = ttime[0]-self.dt*0.5
        t2[-1] = ttime[-1]+self.dt*0.5
        ext = [t2[0], t2[-1], lp[0]-0.5*dp, lp[-1]+0.5*dp]

        if title[1] is None:
            self.axWavelet.set_title('b) Wavelet Power Spectrum')
        else:
            self.axWavelet.set_title(title)
        self.axWavelet.set_ylabel('Period')
        self.axWavelet.minorticks_on()
        self.axWavelet.tick_params(which='both', direction='in')
        self.axWavelet.set_yticks(np.arange(14))
        self.axWavelet.set_yticks(np.log2(np.arange(1,100)), minor=True)
        self.axWavelet.set_yticklabels(2**np.arange(14))
        self.axWavelet.set_ylim(lp[0], lp[-1])

        self.imWavelet = self.axWavelet.imshow(wpower, plt.cm.cubehelix_r, origin='lower', extent=ext, aspect='auto', interpolation='bilinear')
        
        self.axWavelet.contour(t2, lp, sig, [-99,1] ,colors='r')
        self.axWavelet.fill_between(t2, coi, lp.max()+0.5*dp, color='gray', alpha=0.6, zorder=95, hatch='x')

        
        # Plot Global Wavelet Spectrum
        if title[2] is None:
            self.axGlobal.set_title('c) Global')
        else:
            self.axGlobal.set_title(title)
        self.axGlobal.set_xlabel('Power')
        self.axGlobal.set_ylabel('')
        self.axGlobal.minorticks_on()
        self.axGlobal.tick_params(which='both', direction='in')

        self.axGlobal.plot(self.gws, lp, color='k', lw=1.5)
        
        dof = n - self.scale
        gsig = self.waveSignif(self.data, sigtest=1, lag1=lag1, dof=dof, siglvl=siglvl)
        self.axGlobal.plot(gsig, lp,  'r--', lw=1.5)
        
        
        # Plot Scale-average Time Series
        if title[3] is None:
            self.axScaleAvg.set_title('d) Scale-average Time Series')
        else:
            self.axScaleAvg.set_title(title)
        self.axScaleAvg.set_xlabel('Time')
        self.axScaleAvg.set_ylabel('Avg')
        self.axScaleAvg.minorticks_on()
        self.axScaleAvg.tick_params(which='both', direction='in')
        
        nP, nt = self.power.shape
        am = self.power.argmax()
        Pm = am//nt
        tm = am%nt
        wh = sig[:,tm]>=1
        cs = (~wh).cumsum()
        wh2 = cs == cs[tm]
        pp = self.period[wh2]
        pM = pp.max()
        pm = pp.min()
        period_mask = (self.period >= pm)*(self.period <= pM)
        power_norm = self.power/self.scale[:,None]
        power_avg = self.dj*self.dt/self.cdelta*power_norm[period_mask,:].sum(0)
        self.pScaleAvg = self.axScaleAvg.plot(ttime, power_avg, color='k', lw=1.5)
        
        self.fig.tight_layout()
        self.fig.show()

class WaveCoherency:
    def __init__(self, time1, y1, time2, y2, dj=0.1, s0=None, j=None, mother='MORLET', norm=True):
        """
        Compute the wavelet coherency between two time series.
        
        Parameters
        ----------
        y1 : `~numpy.ndarray`
            Input time-series.
        time1 : `~numpy.ndarray`
            time array for time series 1.
        y2 : `~numpy.ndarray`
            Input time-series.
        time2 : `~numpy.ndarray`
            time array for time series 2.
        norm : (optional) bool
            If set to True, normalize the time-series by the standard deviation.
        
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
        self.sig = None
        t1 = time1
        t2 = time2
        dt1 = np.diff(t1).mean()
        dt2 = np.diff(t2).mean()
        ts = max(t1.min(), t2.min())
        te = min(t1.max(), t2.max())
        dt = max(dt1,dt2)
        self.dt = dt
        tnew = np.arange(ts, te, dt)

        f1 = interp1d(t1, y1, kind='cubic')
        f2 = interp1d(t2, y2, kind='cubic')
        y1_new = f1(tnew)
        y2_new = f2(tnew)
        if norm:
            y1_new = (y1_new-y1_new.mean()) / y1_new.std()
            y2_new = (y2_new-y2_new.mean()) / y2_new.std()

        kwargs = dict(dj=dj, j=j, s0=s0, mother=mother)

        wl1 = Wavelet(y1_new, dt, **kwargs)
        wl2 = Wavelet(y2_new, dt, **kwargs)

        self.wl1 = wl1
        self.wl2 = wl2
        self.wl1._motherParam()
        
        scale = wl1.scale

        self.cross_wavelet = wl1.wavelet*wl2.wavelet.conj()
        self.cross_wavelet0 = self.cross_wavelet.copy()
        self.power1 = np.abs(wl1.wavelet)**2
        self.power2 = np.abs(wl2.wavelet)**2
        
        self.time = tnew
        self.scale = scale
        nj = len(self.scale)
        
        s1 = np.abs(wl1.wavelet) **2 / scale[:,None]
        s2 = np.abs(wl2.wavelet) **2 / scale[:,None]
        global1 = self.power1.sum(1)
        global2 = self.power2.sum(1)
        self.global_cross = self.cross_wavelet.sum(1)
        self.global_coher = np.abs(self.global_cross)**2/(global1*global2)
        self.global_phase = np.arctan(self.global_cross.imag/self.global_cross.real)*180./np.pi
        
        # smoothing start
        nt = (4*self.scale/wl1.dt)//2*4+1
        nt2 = nt[:,None]
        ntmax = nt.max()
        g = np.arange(ntmax) * np.ones((nj,1))
        wh = g >= nt2
        time_wavelet = (g-nt2//2)*wl1.dt/self.scale[:,None]
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
        
        nw = int(0.6/wl1.dj/2 + 0.5)*2-1
        weight = np.ones(nw)/nw
        self.cross_wavelet = _fastConv2(self.cross_wavelet, weight)
        self.power1 = _fastConv2(self.power1,weight)
        self.power2 = _fastConv2(self.power2,weight)
        # smoothing end
            
        power3=self.power1*self.power2
        whp=power3 < 1e-9
        power3[whp]=1e-9
        self.wave_coher = (np.abs(self.cross_wavelet)**2/power3).real
        self.wave_phase = np.angle(self.cross_wavelet, deg=True)

    def signif_coh(self, siglvl=0.95, mc_count=1000):
        sig = cal_signif_coh(self.wl1, self.wl2, siglvl=siglvl, mc_count=mc_count)
        self.sig = sig
        return sig
        

    def signif_crossPower(self, siglvl=0.95):
        """
        Compute the significance levels for a cross wavelet spectrum

        Parameters
        ----------
        lag1: (optional) `float`
            LAG 1 Autocorrelation, used for signif levels (default is 0).
        siglvl: (optional) `float`
            Significance level to use (default is 0.95).
            
        Returns
        -------
        signif: `~numpy.ndarray`
            Significance levels as a function of scale.
        """
        y1 = self.wl1.data
        y2 = self.wl2.data
        dt = self.wl1.dt
        freq = self.wl1.freq

        std1 = y1.std()
        std2 = y2.std()
        std = std1*std2

        a1 = ar1(y1)[0]
        a2 = ar1(y2)[0]
        Pk1 = ar1_spectrum(freq * dt, a1)
        Pk2 = ar1_spectrum(freq * dt, a2)
        dof = self.wl1.dofmin
        PPF = chi2.ppf(siglvl, dof)
        signif = (std1 * std2 * (Pk1 * Pk2) ** 0.5 * PPF / dof)
        return signif
    
    def plot(self, interval=1, siglvl=0.95, sigonly=True, color='limegreen', figsize=[8, 4]):
        coh = self.wave_coher
        
        ns, nt = coh.shape
        ph = self.wave_phase
        tstep = max(interval, nt // 30)
        sstep = max(interval, ns // 35)
        if self.sig is None:
            so = self.signif_coh(siglvl, mc_count=1000)
        else:
            so = self.sig
        sig = self.wave_coher/so[:,None]

        coi = self.wl1.coi.copy()
        coi[0] = 1e-3
        coi[-1] = 1e-3
        coi = np.log2(coi)
        lp = np.log2(self.wl1.period)
        roi = (lp[:,None] - coi) < 0
        if sigonly:
            roi = roi * (sig >= 1)
        dp = np.diff(lp).mean()
        ext = [self.time[0]-self.dt*0.5, self.time[-1]+self.dt*0.5, lp[0]-dp*0.5, lp[-1]+dp*0.5]

        X, Y = np.meshgrid(self.time[::tstep], lp[::sstep])
        U = np.cos(np.deg2rad(ph[::sstep, ::tstep]))*roi[::sstep, ::tstep]
        V = np.sin(np.deg2rad(ph[::sstep, ::tstep]))*roi[::sstep, ::tstep]
        wh = U*V != 0
        t2 = self.time.copy()
        t2[0] = self.time[0]-self.dt*0.5
        t2[-1] = self.time[-1]+self.dt*0.5

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(coh, plt.cm.afmhot_r, origin='lower', extent=ext, aspect='auto', interpolation='bilinear', clim=(0.6,1))
        ax.contour(t2, lp, sig, [-99,1], colors='red')
        ax.set_yticks(np.arange(14))
        ax.set_yticks(np.log2(np.arange(1,100)), minor=True)
        ax.set_yticklabels(2**np.arange(14))
        ax.fill_between(t2, coi, lp.max()+dp, color='gray', alpha=0.6, zorder=95, hatch='x')

        ax.quiver(X[wh], Y[wh], U[wh], V[wh], color=color, width=0.006, scale=20, pivot='mid', zorder=10)

        ax.set_ylim(lp[0], lp[-1])

        fig.tight_layout()
        fig.show()
        

def cal_signif_coh(wl1, wl2, siglvl=0.95, mc_count=1000):
    al1 = ar1(wl1.data)[0]
    al2 = ar1(wl2.data)[0]
    dt = wl1.dt
    s0 = wl1.s0
    dj = wl1.dj
    J = wl1.j
    # print('Calculating wavelet coherence significance (Monte Carlo)')
    ms = s0 * (2 ** (J * dj)) / dt
    N = int(np.ceil(ms * 6))
    noise1 = rednoise(N, al1, 1)
    nw = Wavelet(noise1, wl1.dt, dj=wl1.dj, s0=wl1.s0, j=wl1.j, mother=wl1.mother)
    sh = nper, nt = nw.wavelet.shape
    ones = np.ones(sh)
    period = nw.period[:,None] * ones
    coi = nw.coi.copy()
    coi[0] = 1e-3
    coi[-1] = 1e-3
    roi = (period - nw.coi) <= 0
    scales = ones * nw.scale[:,None]
    sig95 = np.zeros(nper)
    maxscale = find(roi.any(axis=1))[-1]
    sig95[roi.any(axis=1)] = np.nan

    nbins = 1000
    wlc = np.ma.zeros([nper, nbins])
    for i in range(mc_count):
        n1 = rednoise(N, al1, 1)
        n2 = rednoise(N, al2, 1)
        nw1 = Wavelet(n1, dt=dt, dj=dj, s0=s0, j=J)
        nw2 = Wavelet(n2, dt=dt, dj=dj, s0=s0, j=J)
        cp, p1, p2 = smoothing(nw1, nw2)
        R2 = np.ma.array(np.abs(cp)**2/(p1*p2), mask=~roi)
        for s in range(maxscale):
            cd = np.floor(R2[s, :] * nbins)
            for j, t in enumerate(cd[~cd.mask]):
                wlc[s, int(t)] += 1

    wlc.mask = (wlc.data == 0.)
    R2y = (np.arange(nbins) + 0.5) / nbins
    for s in range(maxscale):
        sel = ~wlc[s, :].mask
        P = wlc[s, sel].data.cumsum()
        P = (P - 0.5) / P[-1]
        sig95[s] = np.interp(siglvl, P, R2y[sel])

    print('Done')
    return sig95

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

def _chisquareInv(p,v):
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
        x = fmin(_chisquareSolve, minv, maxv, args=(p,v), xtol=tolerance)
        minv = maxv
    x*=v
    return x
    
def _chisquareSolve(xguess,p,v):
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

def ar1(x):
    """
    Allen and Smith autoregressive lag-1 autocorrelation coefficient.
    In an AR(1) model

        x(t) - <x> = \\gamma(x(t-1) - <x>) + \\alpha z(t) ,

    where <x> is the process mean, \\gamma and \\alpha are process
    parameters and z(t) is a Gaussian unit-variance white noise.

    Parameters
    ----------
    x : numpy.ndarray, list
        Univariate time series

    Returns
    -------
    g : float
        Estimate of the lag-one autocorrelation.
    a : float
        Estimate of the noise variance [var(x) ~= a**2/(1-g**2)]
    mu2 : float
        Estimated square on the mean of a finite segment of AR(1)
        noise, mormalized by the process variance.

    References
    ----------
    [1] Allen, M. R. and Smith, L. A. Monte Carlo SSA: detecting
        irregular oscillations in the presence of colored noise.
        *Journal of Climate*, **1996**, 9(12), 3373-3404.
        <http://dx.doi.org/10.1175/1520-0442(1996)009<3373:MCSDIO>2.0.CO;2>
    [2] http://www.madsci.org/posts/archives/may97/864012045.Eg.r.html

    """
    x = np.asarray(x)
    N = x.size
    xm = x.mean()
    x = x - xm

    # Estimates the lag zero and one covariance
    c0 = x.transpose().dot(x) / N
    c1 = x[0:N-1].transpose().dot(x[1:N]) / (N - 1)

    # According to A. Grinsteds' substitutions
    B = -c1 * N - c0 * N**2 - 2 * c0 + 2 * c1 - c1 * N**2 + c0 * N
    A = c0 * N**2
    C = N * (c0 + c1 * N - c1)
    D = B**2 - 4 * A * C

    if D > 0:
        g = (-B - D**0.5) / (2 * A)
    else:
        raise Warning('Cannot place an upperbound on the unbiased AR(1). '
                      'Series is too short or trend is to large.')

    # According to Allen & Smith (1996), footnote 4
    mu2 = -1 / N + (2 / N**2) * ((N - g**N) / (1 - g) -
                                 g * (1 - g**(N - 1)) / (1 - g)**2)
    c0t = c0 / (1 - mu2)
    a = ((1 - g**2) * c0t) ** 0.5

    return g, a, mu2

def ar1_spectrum(freqs, ar1=0.):
    """
    Lag-1 autoregressive theoretical power spectrum.

    Parameters
    ----------
    freqs : numpy.ndarray, list
        Frequencies at which to calculate the theoretical power
        spectrum.
    ar1 : float
        Autoregressive lag-1 correlation coefficient.

    Returns
    -------
    Pk : numpy.ndarray
        Theoretical discrete Fourier power spectrum of noise signal.

    References
    ----------
    [1] http://www.madsci.org/posts/archives/may97/864012045.Eg.r.html

    """

    freqs = np.asarray(freqs)
    Pk = (1 - ar1 ** 2) / np.abs(1 - ar1 * np.exp(-2 * np.pi * 1j * freqs)) \
        ** 2

    return Pk


def rednoise(N, g, a=1.):
    """
    Red noise generator using filter.

    Parameters
    ----------
    N : int
        Length of the desired time series.
    g : float
        Lag-1 autocorrelation coefficient.
    a : float, optional
        Noise innovation variance parameter.

    Returns
    -------
    y : numpy.ndarray
        Red noise time series.

    """
    if g == 0:
        yr = np.randn(N, 1) * a
    else:
        # Twice the decorrelation time.
        tau = int(np.ceil(-2 / np.log(np.abs(g))))
        yr = lfilter([1, 0], [1, -g], np.random.randn(N + tau, 1) * a)
        yr = yr[tau:]

    return yr.flatten()

def find(condition):
    """Returns the indices where ravel(condition) is true."""
    res, = np.nonzero(np.ravel(condition))
    return res

def smoothing(wvl, wvl2):
    nj = len(wvl.scale)
    nt = (4*wvl.scale/wvl.dt)//2*4+1
    nt2 = nt[:,None]
    ntmax = nt.max()
    g = np.arange(ntmax) * np.ones((nj,1))
    wh = g >= nt2
    time_wavelet = (g-nt2//2)*wvl.dt/wvl.scale[:,None]
    wave_func = np.exp(-time_wavelet**2/2)
    wave_func[wh] = 0
    wave_func = (wave_func/wave_func.sum(1)[:,None])
    cp = wvl.wavelet * wvl2.wavelet.conj()
    
    p1 = _fastConv(np.abs(wvl.wavelet)**2, wave_func, nt2)
    p2 = _fastConv(np.abs(wvl2.wavelet)**2, wave_func, nt2)
    cp = _fastConv(cp, wave_func, nt2)
    scales = wvl.scale[:, None]
    cp /= scales
    p1 /= scales
    p2 /= scales

    nw = int(0.6/wvl.dj/2 + 0.5)*2-1
    weight = np.ones(nw)/nw
    cp = np.abs(_fastConv2(cp, weight))
    p1 = np.abs(_fastConv2(p1,weight))
    p2 = np.abs(_fastConv2(p2,weight))

    return cp, p1, p2