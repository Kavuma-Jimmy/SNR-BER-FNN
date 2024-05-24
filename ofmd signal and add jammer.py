import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fft
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter

# OFDM Parameters
num_subcarriers = 64
cp_len = 16  # Cyclic Prefix length
num_symbols = 100  # Number of OFDM symbols

# Generate random OFDM symbols (QPSK Modulation)
def generate_ofdm_symbols(num_subcarriers, num_symbols):
    data = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=(num_symbols, num_subcarriers))
    return data

# Apply IFFT to generate time-domain OFDM signal
def apply_ifft(ofdm_data):
    return np.fft.ifft(ofdm_data, axis=1)

# Add cyclic prefix
def add_cyclic_prefix(ofdm_time, cp_len):
    return np.hstack([ofdm_time[:, -cp_len:], ofdm_time])

# Generate the OFDM signal
ofdm_symbols = generate_ofdm_symbols(num_subcarriers, num_symbols)
ofdm_time = apply_ifft(ofdm_symbols)
ofdm_time_cp = add_cyclic_prefix(ofdm_time, cp_len)

# Simulate Jamming
def add_jammer(signal, jammer_freqs, jammer_power):
    num_samples = signal.shape[1]
    t = np.arange(num_samples)
    jammer_signal = np.zeros_like(signal, dtype=np.complex128)
    for freq in jammer_freqs:
        jammer_signal += jammer_power * np.exp(2j * np.pi * freq * t / num_samples)
    return signal + jammer_signal

jammer_freqs = [10, 20]  # Example jammer frequencies
jammer_power = 0.5
ofdm_time_cp_jammed = add_jammer(ofdm_time_cp, jammer_freqs, jammer_power)

# Plot one OFDM symbol (with cyclic prefix) and its jammed version
plt.figure(figsize=(10, 6))
plt.plot(np.real(ofdm_time_cp[0]), label='Original OFDM Signal')
plt.plot(np.real(ofdm_time_cp_jammed[0]), label='Jammed OFDM Signal')
plt.legend()
plt.title('OFDM Signal with and without Jamming')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.show()

# Design a notch filter to suppress jamming frequencies
def design_notch_filter(jammer_freqs, fs, Q=30):
    b, a = [], []
    for f in jammer_freqs:
        b_i, a_i = butter(2, [f-0.5, f+0.5], btype='bandstop', fs=fs)
        b.append(b_i)
        a.append(a_i)
    return b, a

# Apply the notch filter
def apply_notch_filter(signal, b, a):
    filtered_signal = signal
    for b_i, a_i in zip(b, a):
        filtered_signal = lfilter(b_i, a_i, filtered_signal, axis=1)
    return filtered_signal

# Notch filter parameters
fs = num_subcarriers + cp_len  # Sample rate
b, a = design_notch_filter(jammer_freqs, fs)

# Apply the notch filter before IFFT
filtered_ofdm_time_cp_jammed = apply_notch_filter(ofdm_time_cp_jammed, b, a)

# Plot the filtered signal
plt.figure(figsize=(10, 6))
plt.plot(np.real(filtered_ofdm_time_cp_jammed[0]), label='Filtered OFDM Signal')
plt.legend()
plt.title('Filtered OFDM Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.show()

# Remove cyclic prefix
def remove_cyclic_prefix(signal, cp_len):
    return signal[:, cp_len:]

# FFT to go back to frequency domain
def apply_fft(ofdm_time):
    return np.fft.fft(ofdm_time, axis=1)

# Remove cyclic prefix from the received signal
ofdm_time_cp_jammed_no_cp = remove_cyclic_prefix(ofdm_time_cp_jammed, cp_len)
filtered_ofdm_time_cp_jammed_no_cp = remove_cyclic_prefix(filtered_ofdm_time_cp_jammed, cp_len)

# Apply FFT to get back to the frequency domain
ofdm_freq_jammed = apply_fft(ofdm_time_cp_jammed_no_cp)
filtered_ofdm_freq_jammed = apply_fft(filtered_ofdm_time_cp_jammed_no_cp)

# Detect jammers by finding peaks in the frequency domain
def detect_jammers(ofdm_freq, threshold=0.5):
    power_spectrum = np.abs(ofdm_freq)**2
    jammer_indices = np.where(power_spectrum.mean(axis=0) > threshold)[0]
    return jammer_indices

jammer_indices = detect_jammers(ofdm_freq_jammed)

# Suppress detected jammers
def suppress_jammers(ofdm_freq, jammer_indices):
    ofdm_freq[:, jammer_indices] = 0
    return ofdm_freq

suppressed_ofdm_freq = suppress_jammers(filtered_ofdm_freq_jammed, jammer_indices)

# Plot the power spectrum before and after suppression
plt.figure(figsize=(10, 6))
plt.plot(np.abs(ofdm_freq_jammed.mean(axis=0)), label='Jammed Signal Power Spectrum')
plt.plot(np.abs(suppressed_ofdm_freq.mean(axis=0)), label='Suppressed Signal Power Spectrum')
plt.legend()
plt.title('Power Spectrum Before and After Jammer Suppression')
plt.xlabel('Subcarrier Index')
plt.ylabel('Power')
plt.show()

# Prepare data for neural network (flatten the data)
X = np.abs(filtered_ofdm_freq_jammed)
y = (np.max(X, axis=1) > 0.5).astype(int)  # Example target: 1 if jammer present, 0 otherwise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Flatten(input_shape=(num_subcarriers,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Use the model to detect and suppress jammers
jammer_probs = model.predict(X)
suppressed_ofdm_freq_nn = filtered_ofdm_freq_jammed.copy()
suppressed_ofdm_freq_nn[jammer_probs.flatten() > 0.5] = 0

# Plot the power spectrum after neural network suppression
plt.figure(figsize=(10, 6))
plt.plot(np.abs(filtered_ofdm_freq_jammed.mean(axis=0)), label='Filtered Signal Power Spectrum')
plt.plot(np.abs(suppressed_ofdm_freq_nn.mean(axis=0)), label='NN Suppressed Signal Power Spectrum')
plt.legend()
plt.title('Power Spectrum After Neural Network Suppression')
plt.xlabel('Subcarrier Index')
plt.ylabel('Power')
plt.show()
