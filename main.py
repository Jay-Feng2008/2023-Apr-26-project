import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from scipy.io.wavfile import read, write
from scipy.fft import fft, ifft, fftfreq
from scipy import signal

def read_wav(file_name):
    a = read(file_name)
    return np.array(a[1], dtype=np.float64)

def fourier(num_sample_points=0, sample_spacing=0, y=[], inverse=False):
    if inverse == 0:
        # sample spacing
        x = np.linspace(0.0, num_sample_points * sample_spacing, num_sample_points, endpoint=False)
        yf = fft(y)
        xf = fftfreq(num_sample_points, sample_spacing)[:num_sample_points//2]
        result = (xf, 2.0/num_sample_points * np.abs(yf[0:num_sample_points//2]), yf)
        return result


calibration_tone = read_wav("calibration_tone.wav")     # The signal of the tuning fork
impulse = read_wav("impulse.wav")       # signal of impulse for convolution
calibration_impulse = read_wav("calibration_impulse.wav")       # signal of the ruler striking the tuning fork

calibration_tone_fft = fourier(len(calibration_tone), 0.36146/len(calibration_tone), calibration_tone)      # lasts 0.36146 seconds
impulse_fft = fourier(len(impulse), 0.20986/len(impulse), impulse)      # lasts 0.20986 seconds
calibration_impulse_fft = fourier(len(calibration_impulse), 0.100210/len(calibration_impulse), calibration_impulse)     # lasts 0.100210 seconds


plt.plot(calibration_tone_fft[0], calibration_tone_fft[1])
plt.plot(impulse_fft[0], impulse_fft[1])
plt.plot(calibration_impulse_fft[0], calibration_impulse_fft[1])
plt.show()

# from the plot, we see the tuning fork has freq peaks:
# 1st peak: 439.88
# 2nd peak: 2749.95
# 3rd peak: 4822.110
# 4th peak: 4824.876
# 5th peak: 5120.898

# We use band-stop filter to filter these freqs out
# This leaves the freq peaks of the ruler vibrating

freq_peaks = [440, 2749.95, 4822.110, 4824.876, 5120.898]

def band_stop_filtering(sample_freq=0, freq_peaks=[], Q=30.0, input_signal=[]):
    for freq in freq_peaks:
        b, a = signal.iirnotch(freq, Q, sample_freq)
        input_signal = signal.filtfilt(b, a, input_signal)
    return input_signal

filtered = band_stop_filtering((0.100210/len(calibration_impulse))**(-1), freq_peaks, 10, calibration_impulse)

filtered_fft = fourier(len(filtered), 0.100210/len(filtered), filtered)
plt.plot(filtered_fft[0], filtered_fft[1])
plt.show()

# Looking at the plot, we see new peaks:
# 1238.850

# We now filter this freq on the impulse signal which will be used for final convolution
filtered_final_impluse = band_stop_filtering((0.20986/len(impulse))**(-1), [1238.850], 20, impulse)
filtered_final_impluse_fft = fourier(len(filtered_final_impluse), 0.20986/len(filtered_final_impluse), filtered_final_impluse)
plt.plot(filtered_final_impluse_fft[0], filtered_final_impluse_fft[1])
plt.show()

# Save the filtered impulse for listening.
scaled = np.int16(filtered_final_impluse / np.max(np.abs(filtered_final_impluse)) * 32767)
write('filtered_impulse.wav', int((0.20986/len(impulse))**(-1)), scaled)


## Finally, we do convolution between a music piece and the filtered impulse
music = read_wav("music.wav")
conv = np.convolve(filtered_final_impluse[0:20], music, 'same')
scaled = np.int16(conv / np.max(np.abs(conv)) * 32767)
write('convovled_music.wav', int(len(conv)/14.08915), scaled)              # The music lasts for 14 seconds
