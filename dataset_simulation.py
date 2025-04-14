import numpy as np
import librosa
import os
import soundfile as sf
import matplotlib.pyplot as plt
# from scipy import signal
from scipy.signal.windows import gaussian
import random

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def simulate_normal_sound(duration=3.0, sr=22050):
    """
    Simulate normal equipment sound
    - Steady frequency components
    - Low noise
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Base frequency components (e.g., motor rotation)
    base_freq = random.uniform(40, 60)  # Base frequency in Hz
    
    # Create signal with fundamental frequency and harmonics
    audio = 0.5 * np.sin(2 * np.pi * base_freq * t)
    audio += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
    audio += 0.15 * np.sin(2 * np.pi * base_freq * 3 * t)
    audio += 0.1 * np.sin(2 * np.pi * base_freq * 4 * t)
    
    # Add background noise (low level)
    noise = np.random.normal(0, 0.05, len(t))
    audio += noise
    
    # Add small random amplitude modulation
    am_freq = random.uniform(0.2, 1.0)
    am_depth = random.uniform(0.02, 0.1)
    am = 1 + am_depth * np.sin(2 * np.pi * am_freq * t)
    audio *= am
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio, sr

def simulate_early_fault_sound(duration=3.0, sr=22050):
    """
    Simulate early fault equipment sound
    - Same base frequency as normal
    - Additional intermittent components
    - Slightly higher noise
    - Small periodic pulse (early bearing fault)
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Base frequency components (similar to normal)
    base_freq = random.uniform(40, 60)
    
    # Create signal with fundamental frequency and harmonics
    audio = 0.5 * np.sin(2 * np.pi * base_freq * t)
    audio += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
    audio += 0.15 * np.sin(2 * np.pi * base_freq * 3 * t)
    audio += 0.1 * np.sin(2 * np.pi * base_freq * 4 * t)
    
    # Add background noise (medium level)
    noise = np.random.normal(0, 0.1, len(t))
    audio += noise
    
    # Add intermittent high-frequency component (early fault indicator)
    fault_freq = random.uniform(500, 2000)  # Fault frequency
    fault_strength = random.uniform(0.05, 0.2)
    fault_intervals = np.random.random(len(t)) < 0.2  # Occurs ~20% of the time
    fault_component = fault_strength * np.sin(2 * np.pi * fault_freq * t) * fault_intervals
    audio += fault_component
    
    # Add small periodic pulse (simulating early bearing fault)
    pulse_rate = random.uniform(8, 15)  # Pulses per second
    pulse_width = int(sr * 0.01)  # 10ms pulse width
    num_pulses = int(duration * pulse_rate)
    
    for i in range(num_pulses):
        pulse_time = random.uniform(0, duration)
        pulse_pos = int(pulse_time * sr)
        if pulse_pos + pulse_width < len(audio):
            pulse_shape = gaussian(pulse_width, std=pulse_width/6)
            pulse_strength = random.uniform(0.05, 0.15)
            audio[pulse_pos:pulse_pos+pulse_width] += pulse_shape * pulse_strength
    
    # Add amplitude modulation
    am_freq = random.uniform(0.2, 1.0)
    am_depth = random.uniform(0.1, 0.2)
    am = 1 + am_depth * np.sin(2 * np.pi * am_freq * t)
    audio *= am
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio, sr

def simulate_failure_sound(duration=3.0, sr=22050):
    """
    Simulate failure equipment sound
    - Irregular frequency components
    - High noise level
    - Strong irregular pulses
    - Possible frequency shifts
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Base frequency components with instability
    base_freq = random.uniform(35, 65)  # More variation than normal
    
    # Create unstable signal with fundamental frequency and harmonics
    # Add frequency wobble to simulate instability
    wobble = 0.05 * np.sin(2 * np.pi * 0.5 * t)
    
    audio = 0.4 * np.sin(2 * np.pi * base_freq * (1 + wobble) * t)
    audio += 0.25 * np.sin(2 * np.pi * base_freq * 2 * (1 + wobble) * t)
    audio += 0.15 * np.sin(2 * np.pi * base_freq * 3 * (1 + wobble) * t)
    audio += 0.1 * np.sin(2 * np.pi * base_freq * 4 * (1 + wobble) * t)
    
    # Add strong background noise
    noise = np.random.normal(0, 0.2, len(t))
    audio += noise
    
    # Add stronger high-frequency components
    for _ in range(3):
        fault_freq = random.uniform(1000, 4000)
        fault_strength = random.uniform(0.15, 0.3)
        audio += fault_strength * np.sin(2 * np.pi * fault_freq * t)
    
    # Add strong irregular pulses (simulating severe faults)
    pulse_rate = random.uniform(15, 30)  # More pulses per second
    pulse_width = int(sr * 0.02)  # 20ms pulse width
    num_pulses = int(duration * pulse_rate)
    
    for i in range(num_pulses):
        pulse_time = random.uniform(0, duration)
        pulse_pos = int(pulse_time * sr)
        if pulse_pos + pulse_width < len(audio):
            pulse_shape = gaussian(pulse_width, std=pulse_width/5)
            pulse_strength = random.uniform(0.2, 0.5)
            audio[pulse_pos:pulse_pos+pulse_width] += pulse_shape * pulse_strength
    
    # Add strong amplitude modulation
    am_freq = random.uniform(0.5, 2.0)
    am_depth = random.uniform(0.2, 0.4)
    am = 1 + am_depth * np.sin(2 * np.pi * am_freq * t)
    audio *= am
    
    # Add occasional dropouts (equipment stopping/starting)
    dropout_points = np.random.random(len(t)) < 0.05
    audio[dropout_points] *= 0.2
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    return audio, sr

def add_environmental_noise(audio, noise_level=0.05):
    """Add environmental noise to audio"""
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise

def visualize_examples(normal, early_fault, failure, sr):
    """Visualize example waveforms and spectrograms"""
    plt.figure(figsize=(15, 12))
    
    # Waveforms
    plt.subplot(3, 2, 1)
    plt.title('Normal Equipment - Waveform')
    librosa.display.waveshow(normal, sr=sr)
    
    plt.subplot(3, 2, 3)
    plt.title('Early Fault - Waveform')
    librosa.display.waveshow(early_fault, sr=sr)
    
    plt.subplot(3, 2, 5)
    plt.title('Failure - Waveform')
    librosa.display.waveshow(failure, sr=sr)
    
    # Spectrograms
    plt.subplot(3, 2, 2)
    plt.title('Normal Equipment - Spectrogram')
    D = librosa.amplitude_to_db(np.abs(librosa.stft(normal)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 2, 4)
    plt.title('Early Fault - Spectrogram')
    D = librosa.amplitude_to_db(np.abs(librosa.stft(early_fault)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 2, 6)
    plt.title('Failure - Spectrogram')
    D = librosa.amplitude_to_db(np.abs(librosa.stft(failure)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()

def generate_dataset(output_dir, num_samples=50, sr=22050):
    """Generate a complete dataset of simulated equipment sounds"""
    # Create directory structure
    create_directory(output_dir)
    create_directory(os.path.join(output_dir, "normal"))
    create_directory(os.path.join(output_dir, "early_fault"))
    create_directory(os.path.join(output_dir, "failure"))
    
    # Generate normal equipment sounds
    print(f"Generating {num_samples} normal equipment sound samples...")
    for i in range(num_samples):
        # Variable duration between 2-5 seconds
        duration = random.uniform(2.0, 5.0)
        audio, sr = simulate_normal_sound(duration, sr)
        
        # Add some environmental noise
        audio = add_environmental_noise(audio, noise_level=random.uniform(0.01, 0.05))
        
        # Save the file
        filename = os.path.join(output_dir, "normal", f"normal_{i+1:03d}.wav")
        sf.write(filename, audio, sr)
        
        if i % 10 == 0:
            print(f"  Generated {i+1}/{num_samples} normal samples")
    
    # Generate early fault equipment sounds
    print(f"Generating {num_samples} early fault equipment sound samples...")
    for i in range(num_samples):
        duration = random.uniform(2.0, 5.0)
        audio, sr = simulate_early_fault_sound(duration, sr)
        
        # Add some environmental noise
        audio = add_environmental_noise(audio, noise_level=random.uniform(0.01, 0.05))
        
        # Save the file
        filename = os.path.join(output_dir, "early_fault", f"early_fault_{i+1:03d}.wav")
        sf.write(filename, audio, sr)
        
        if i % 10 == 0:
            print(f"  Generated {i+1}/{num_samples} early fault samples")
    
    # Generate failure equipment sounds
    print(f"Generating {num_samples} failure equipment sound samples...")
    for i in range(num_samples):
        duration = random.uniform(2.0, 5.0)
        audio, sr = simulate_failure_sound(duration, sr)
        
        # Add some environmental noise
        audio = add_environmental_noise(audio, noise_level=random.uniform(0.01, 0.05))
        
        # Save the file
        filename = os.path.join(output_dir, "failure", f"failure_{i+1:03d}.wav")
        sf.write(filename, audio, sr)
        
        if i % 10 == 0:
            print(f"  Generated {i+1}/{num_samples} failure samples")
    
    print(f"\nDataset generation complete. Files saved to {output_dir}")
    
    # Generate and display example visualizations
    normal, sr = simulate_normal_sound()
    early_fault, _ = simulate_early_fault_sound()
    failure, _ = simulate_failure_sound()
    
    visualize_examples(normal, early_fault, failure, sr)
    
    return output_dir

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Output directory for the dataset
    output_dir = "equipment_sound_dataset"
    
    # Number of samples per class
    num_samples = 50
    
    # Generate the dataset
    dataset_path = generate_dataset(output_dir, num_samples)
    
    print(f"""
    Dataset generated successfully!
    
    Location: {os.path.abspath(dataset_path)}
    Classes: normal, early_fault, failure
    Samples per class: {num_samples}
    
    You can now use this dataset with the SVM classification model.
    """)