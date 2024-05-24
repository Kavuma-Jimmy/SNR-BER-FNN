import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split

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

# Remove cyclic prefix
def remove_cyclic_prefix(signal, cp_len):
    return signal[:, cp_len:]

# FFT to go back to frequency domain
def apply_fft(ofdm_time):
    return np.fft.fft(ofdm_time, axis=1)

# Apply FFT to the jammed signal
ofdm_time_cp_jammed_no_cp = remove_cyclic_prefix(ofdm_time_cp_jammed, cp_len)
ofdm_freq_jammed = apply_fft(ofdm_time_cp_jammed_no_cp)

# Pre-FFT Jammer Suppression (Notch Filter in Time Domain)
def design_notch_filter(jammer_freqs, fs, Q=30):
    b, a = [], []
    for f in jammer_freqs:
        b_i, a_i = butter(2, [f-0.5, f+0.5], btype='bandstop', fs=fs)
        b.append(b_i)
        a.append(a_i)
    return b, a

def apply_notch_filter(signal, b, a):
    filtered_signal = signal
    for b_i, a_i in zip(b, a):
        filtered_signal = lfilter(b_i, a_i, filtered_signal, axis=1)
    return filtered_signal

fs = num_subcarriers + cp_len  # Sample rate
b, a = design_notch_filter(jammer_freqs, fs)
pre_fft_filtered_signal = apply_notch_filter(ofdm_time_cp_jammed, b, a)

# Convert pre-FFT filtered signal to frequency domain
pre_fft_filtered_signal_no_cp = remove_cyclic_prefix(pre_fft_filtered_signal, cp_len)
pre_fft_filtered_freq = apply_fft(pre_fft_filtered_signal_no_cp)

# Post-FFT Jammer Suppression (Notch Filter in Frequency Domain)
def detect_jammers(ofdm_freq, threshold=0.5):
    power_spectrum = np.abs(ofdm_freq)**2
    jammer_indices = np.where(power_spectrum.mean(axis=0) > threshold)[0]
    return jammer_indices

jammer_indices = detect_jammers(ofdm_freq_jammed)

def suppress_jammers(ofdm_freq, jammer_indices):
    ofdm_freq[:, jammer_indices] = 0
    return ofdm_freq

post_fft_suppressed_freq = suppress_jammers(ofdm_freq_jammed.copy(), jammer_indices)

# Prepare data for FNN
X = np.abs(ofdm_freq_jammed)
y = (np.max(X, axis=1) > 0.5).astype(int)  # Example target: 1 if jammer present, 0 otherwise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
STEP 3: CREATE MODEL CLASS
'''
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out

'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = num_subcarriers
hidden_dim = 128
output_dim = 1

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.BCEWithLogitsLoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

'''
STEP 7: TRAIN THE MODEL
'''
num_epochs = 20
batch_size = 32

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        # Clear gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # Update parameters
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    

# Evaluate the model
with torch.no_grad():
    model.eval()
    outputs = model(X_test_tensor)
    predicted = (torch.sigmoid(outputs).numpy() > 0.5).astype(int)
    accuracy = np.mean(predicted == y_test.reshape(-1, 1))  # No need for .numpy() here
    print(f'Test Accuracy: {accuracy:.4f}')

# Apply FNN-based jammer suppression
jammer_probs = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).detach().numpy()
nn_suppressed_freq = ofdm_freq_jammed.copy()
nn_suppressed_freq[jammer_probs.flatten() > 0.5] = 0

# Plot Results
def plot_signal(signal, title, ylabel='BER (Hz)'):
    plt.figure(figsize=(10, 6))
    plt.plot(np.real(signal[0]), label='desired signal')
    plt.plot(np.imag(signal[0]), label='jamming signal')
    plt.legend()
    plt.title(title)
    plt.xlabel('SNR (dB)')
    plt.ylabel(ylabel)
    plt.show()

def plot_power_spectrum(ofdm_freq, title):
    plt.figure(figsize=(10, 6))
    plt.plot(np.abs(ofdm_freq.mean(axis=0)), label='Power Spectrum')
    plt.legend()
    plt.title(title)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER (Hz)')
    plt.show()

plot_signal(pre_fft_filtered_signal, 'Pre-FFT Suppression (Time Domain)')
plot_power_spectrum(pre_fft_filtered_freq, 'Pre-FFT Filtered (Frequency Domain)')

# Plotting Post-FFT Suppression with two lines from the Pre-FFT signal
plt.figure(figsize=(10, 6))
plt.plot(np.abs(ofdm_freq_jammed.mean(axis=0)), label='Pre-FFT Jammed Signal Power Spectrum')
plt.plot(np.abs(pre_fft_filtered_freq.mean(axis=0)), label='Pre-FFT Filtered Power Spectrum', linestyle='--')
plt.plot(np.abs(post_fft_suppressed_freq.mean(axis=0)), label='Post-FFT Suppression Power Spectrum', linestyle='-.')
plt.legend()
plt.title('Post-FFT Suppression')
plt.xlabel('SNR (dB)')
plt.ylabel('BER (Hz)')
plt.show()

plot_power_spectrum(nn_suppressed_freq, 'Feed-Forward Neural Network Suppression (Frequency Domain)')

plot_signal(pre_fft_filtered_signal, 'Pre-IFFT Filtered (Time Domain)')
plot_signal(pre_fft_filtered_signal, 'Post-IFFT Filtered (Time Domain)')

