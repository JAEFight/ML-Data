import os
import numpy as np
import librosa
import soundfile as sf
paths = "d:/Speech_recognition/not_preprocessed_files/weather"
labels=os.listdir(paths)
#find count of each label and plot bar graph

def change_krz():
    for label in labels:
        # Ein loop mit dem man bedingungen machen kann
        waves = []
        for wav in os.listdir(paths + '/'+ label):
            waves.append(wav)
        new_extension = '.wav'
        for f in waves:
            f = paths+'/'+label+'/'+f
            pre, ext = os.path.splitext(f)
            os.rename(f, pre + new_extension)
change_krz()


def process():
    file_path = "D:/Speech_recognition/not_preprocessed_files/weather"
    sample_rate = 8000
    target_duration = 1.0
    # Lade die Datei und stelle sicher, dass sie auf die gewünschte Samplingrate konvertiert wird
    samples, sr = librosa.load(file_path, sr=sample_rate)

    # Zielanzahl der Samples für 1 Sekunde
    target_samples = int(target_duration * sample_rate)
    
    # Finde den Bereich mit der höchsten Energie (lauter Bereich) im Signal
    energy = np.abs(samples)**2  # Energie ist das Quadrat der Amplitude
    window_size = target_samples  # Fenstergröße von 1 Sekunde
    energy_sum = np.convolve(energy, np.ones(window_size), mode='valid')  # Energie über Fenster summieren
    max_energy_idx = np.argmax(energy_sum)  # Startindex des Fensters mit der höchsten Energie
    
    # Bestimme den Start- und Endpunkt der gewünschten Sekunde
    start_idx = max(max_energy_idx, 0)
    end_idx = start_idx + target_samples
    
    # Schneide oder fülle das Signal so, dass es genau 1 Sekunde lang ist
    if len(samples) >= target_samples:
        # Schneiden
        processed_samples = samples[start_idx:end_idx]
    else:
        # Auffüllen mit Nullen, falls die Datei kürzer als 1 Sekunde ist
        processed_samples = np.zeros(target_samples)
        start_fill_idx = (target_samples - len(samples)) // 2
        processed_samples[start_fill_idx:start_fill_idx + len(samples)] = samples
    return processed_samples, sr

output_folder = 'D:/Speech_recognition/not_preprocessed_files/weather/check'
input_folder = paths
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.wav'):
        file_path = os.path.join(input_folder, filename)
        
        # Verarbeite die Audiodatei
        processed_samples, sr = process(file_path, target_duration=1.0, sample_rate=8000)
        
        # Speichere die verarbeitete Datei
        output_path = os.path.join(output_folder, filename)
        sf.write(output_path, processed_samples, sr)
        print(f"Processed and saved: {output_path}")