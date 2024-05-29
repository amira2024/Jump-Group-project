# importing the libraries to filter
import os
import math
import numpy as np
from scipy.signal import butter, freqz, filtfilt, firwin, iirnotch, lfilter, find_peaks


# collection of csv files to filter
csv_files = ["data/Accelerometer_GoodJump1.csv",
             "data/Accelerometer_GoodJump2.csv",
             "data/Accelerometer_GoodJump3.csv",
             "data/Accelerometer_BadJump1.csv",
             "data/Accelerometer_BadJump2.csv",
             "data/Accelerometer_Standing1.csv",
             "data/Accelerometer_Standing2.csv"
             ]

#Using butterworth to filter the data 
order = 4
fs = 50.0  # sample rate, Hz
cutoff = 2.7 # desired cutoff frequency of the filter, Hz. MODIFY AS APPROPROATE


nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
b, a = butter(order, normal_cutoff, btype='low', analog=False)

#filtered_signal = filtfilt(b, a, signal)

# iterate through the csv files
for data_file in csv_files:
    raw_data = np.loadtxt(data_file, delimiter=',', unpack=True, skiprows=1)
    (timestamps, seconds_passed, z_data, y_data, x_data) = raw_data

    signal = np.sqrt(x_data**2 + y_data**2 + z_data**2)

    filtered_signal_x = filtfilt(b, a, x_data)
    filtered_signal_y = filtfilt(b, a, y_data)
    filtered_signal_z = filtfilt(b, a, z_data)

    filtered_data = np.column_stack((timestamps, seconds_passed, filtered_signal_x, filtered_signal_y, filtered_signal_z))

    output_file = "filtered-data/filtered-" + data_file.split('_')[1]

    np.savetxt(output_file, filtered_data, delimiter = ",")

