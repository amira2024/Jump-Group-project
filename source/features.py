# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
from scipy.signal import find_peaks


def _compute_magnitude_features(accel_x, accel_y, accel_z):
    """
    Computes the magnitude of the signal from the x, y, and z axis
    """
    accel_mag = np.sqrt(np.array(accel_x)**2 + np.array(accel_y)**2 + np.array(accel_z)**2)
    return accel_mag # type: ndarray
    
def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(window, axis=0) # type: float

def _compute_variance_features(window):
    """
    Computing the variance of the data in the window
    """
    return np.var(window, axis=0) # type: float

    
def _compute_fft_features(window):
    """
    Computes the dominant frequencies using Discrete Fourier Transform
    """
    feature = np.abs(np.fft.rfft(window, axis=0).astype(float)) # ndarray
    return len(feature)

def _compute_entropy_features(signal):
    # calculate entropy of raw signal
    entrop_mag = _compute_magnitude_features(signal[:,0], signal[:,1], signal[:,2])
    # calculated FFT of signal
    # calculated entropy of FFT
    fft_entropy = np.histogram(entrop_mag,bins = ([0,1,2,3]))[0] 
    
    return sum(fft_entropy) # This is a list of entropy features of raw signal and fft_entropy is  entropy of the signal's FFT.

def _compute_peak_features(window):
    """
    Computes the peaks from signal in the window
    """
    peaks, _ = find_peaks(window, height=35)
    # peaks is an ndarray
    return len(peaks) # returns number of peaks, type: float

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """

    """
    Statistical
    These include the mean, variance and the rate of zero- or mean-crossings. The
    minimum and maximum may be useful, as might the median
    
    FFT features
    use rfft() to get Discrete Fourier Transform
    
    Entropy
    Integrating acceleration
    
    Peak Features:
    Sometimes the count or location of peaks or troughs in the accelerometer signal can be
    an indicator of the type of activity being performed. This is basically what you did in
    assignment A1 to detect steps. Use the peak count over each window as a feature. Or
    try something like the average duration between peaks in a window.
    """

    
    x = []
    feature_names = []
    win = np.array(window)
    x.append(_compute_mean_features(win[:,0]))
    feature_names.append("x_mean")

    x.append(_compute_mean_features(win[:,1]))
    feature_names.append("y_mean")

    x.append(_compute_mean_features(win[:,2]))
    feature_names.append("z_mean")
    
    # fft
    fft_x = _compute_fft_features(win[:,0])
    fft_y = _compute_fft_features(win[:,1])
    fft_z = _compute_fft_features(win[:,2])
    x.append(np.sqrt(fft_x **2 + fft_y **2 + fft_z **2))
    feature_names.append("fft_mag")
    
    # variance
    x.append(_compute_variance_features(win[:,0]))
    feature_names.append("x_variance")
    
    x.append(_compute_variance_features(win[:,1]))
    feature_names.append("y_variance")

    x.append(_compute_variance_features(win[:,2]))
    feature_names.append("z_variance")
    
    # entropy
    x.append(_compute_entropy_features(win))
    # entropy_x = _compute_entropy_features(win[:,0])
    # entropy_y = _compute_entropy_features(win[:,1])
    # entropy_z = _compute_entropy_features(win[:,2])
    # x.append(np.sqrt(entropy_x**2 + entropy_y**2 + entropy_z**2))
    feature_names.append("entropy_mag")
    
    # peaks
    x.append(
        _compute_peak_features(
            _compute_magnitude_features(win[:,0], win[:,1], win[:,2])
        )
    )
    feature_names.append("accel_peaks")

    
    feature_vector = list(x)
    return feature_names, feature_vector