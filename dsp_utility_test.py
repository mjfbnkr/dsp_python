# utility functions for digital signal processing in python
import numpy as np

def calc_fft(x, fs=1e4, average_fft = False):
    """
    Function for estimating fft of n signals. Estimation 

    Parameters
    ----------
    x : array_like
        input signal of shape (n_measurement, n_observation)
    fs : integer
        Sampling frequency of input- and output signal
    average_fft : boolean
        Compute and return the average fft over all n_measurement. 
        Default is False.
        

    Returns
    -------
    f
        array with frequencies
    
    fft/fft_mean
        array with (mean) fft values. If x is one-dimensional and average is set True, mean fft equals fft.
    
    N
        number of sampling points


    Notes
    ------
    None
    """
    from scipy.fftpack import fft
    
    # make sure input dimensions match requirements
    if len(x.shape) == 1:
        x = x[np.newaxis,:]
        single_input=True
    else:
        single_input=False
    N = x.shape[1]
    T = 1 / fs # sample spacing
    f = np.linspace(0.0, 1.0/(2.0*T), N//2)
    
    # calculate fft and mean
    fft_ = []
    for x_ in x:
        fft_.append(fft(x_))
    fft_ = np.array(fft_)
    
    # return mean/single_input fft
    if (average_fft == True) | (single_input):
        fft_mean = np.mean(fft_, axis=0)
        return f, fft_mean, N
    #return several ffts in a 2d array
    else:
        return f, fft_, N


def calc_csd(x,y,fs=1e4):
    """
    Function for calculating cross spectral density of input and output signals.
    If input and output is the same, the spectral density of that signal is calculated.

    Parameters
    ----------
    x : array_like
        input signal of shape (n_measurement, n_observation)
    y : array_like
        output signal of shape (n_measurement, n_observation)
    fs: integer
        Sampling frequency of input- and output signal

    Returns
    -------
    f
        array with frequencies
    
    csd
        array with cross spectral density values 

    Notes
    ------
    None
    """       
    # Calculate fft of input and output signals
    f, xfft, Nx = calc_fft(x, fs=fs, average_fft=False)
    f, yfft, Ny = calc_fft(y, fs=fs, average_fft=False)

    # calculate cross spectral density
    if len(x.shape) > 1:
        csd = []
        for x_, y_ in zip(xfft,yfft):
            csd.append(2/Nx*np.conj(x_[:Nx//2])*2/Ny*y_[:Ny//2])
        csd = np.array(csd)
    else:
        csd = np.conj(2/Nx*xfft[:Nx//2])*2/Ny*yfft[:Ny//2]
    return f, csd
    
    
def calc_coher(x, y, fs=1e4):
    """
    Function for calculating coherence of at least two input and output signals.

    Parameters
    ----------
    x : array_like
        input of shape (n_measurement, n_observation)
    y : array_like
        output of shape (n_measurement, n_observation)
    fs: integer
        Sampling frequency of input- and output signals

    Returns
    -------
    f
        array with frequencies
    
    coher
        array with coherence values 

    Notes
    ------
    None
    """
    if len(x)!=len(y):
        raise ValueError('Length x must be equal length y!')
    elif len(x.shape) < 2:
        raise ValueError('At least two measurements are needed!')
        return
    else:
        # Calculate f
        N = max(x.shape) # number of sample points
        T = 1 / fs # sample spacing
        f = np.linspace(0.0, 1.0/(2.0*T), N//2)

    # calculate cross spectral density and mean
    csd_ = []
    for x_, y_ in zip(x,y):
        csd_.append(calc_csd(x_,y_, fs=fs)[1])
    csd_ = np.array(csd_)
    csd_mean = np.mean(csd_, axis=0)
         
    # calculate auto spectral densities and mean
    asd_x = []
    asd_y = []
    for x_, y_ in zip(x,y):
        asd_x.append(calc_csd(x_,x_, fs=fs)[1])
        asd_y.append(calc_csd(y_,y_, fs=fs)[1])
    asd_x = np.array(asd_x)
    asd_x_mean = np.mean(asd_x, axis=0)
    asd_y = np.array(asd_y)
    asd_y_mean = np.mean(asd_y, axis=0)
        
    #calculate coherence
    coher = np.abs(csd_mean)**2/asd_x_mean/asd_y_mean
        
    return f, coher


def calc_transfer(x, y, fs=1e4, method='H3', return_std=True):
    """
    Function for calculating transfer function of at least two input and output signals.

    Parameters
    ----------
    x : array_like
        input of shape (n_measurement, n_observation)
    y : array_like
        output of shape (n_measurement, n_observation)
    fs: integer
        Sampling frequency of input- and output signals
    method : string
        Select between H1 (noise on the output only), H2 (noise on the input only) and H3 (geometric mean of H1 and H2) method for calculating the coherence. Default is H3.

    Returns
    -------
    f
        array with frequencies
    
    G
        array with transfer function values 

    Notes
    ------
    Assuming sample points N > measurements M -> otherwise change code (!)
    """
    N = max(x.shape) # number of sample points
    T = 1 / fs # sample spacing
    f = np.linspace(0.0, 1.0/(2.0*T), N//2)
    
    if method == 'H3':
        Syx = np.mean(calc_csd(y, x, fs=fs)[1], axis=0)
        Sxx = np.mean(calc_csd(x, x, fs=fs)[1], axis=0)
        Syy = np.mean(calc_csd(y, y, fs=fs)[1], axis=0)

        G1 = Syx/Sxx
        G2 = Syy/Syx
        G = np.sqrt(G1*G2)
    elif method == 'H1':
        Syx = np.mean(calc_csd(y, x, fs=fs)[1], axis=0)
        Sxx = np.mean(calc_csd(x, x, fs=fs)[1], axis=0)
        G = Syx/Sxx
    elif method == 'H2':
        Syy = np.mean(calc_csd(y, y, fs=fs)[1], axis=0)
        Syx = np.mean(calc_csd(y, x, fs=fs)[1], axis=0)
        G = Syy/Syx
    
    coher = calc_coher(x,y, fs=fs)[1]
    std = (np.sqrt(1-coher**2)/(np.abs(coher)*np.sqrt(2*min(x.shape))))*np.abs(G)
    
    return f, G, std