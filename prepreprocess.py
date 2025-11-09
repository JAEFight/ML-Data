import os
import numpy as np
import librosa
import soundfile as sf

wort = "check"
paths = f"D:/Speech_recognition_files/files1"
#paths = f"D:/Speech_recognition/not_preprocessed_files/{wort}"
labels=os.listdir(paths)


def change_krz():
    for label in labels:
        # Ein loop mit dem man bedingungen machen kann
        #waves = [f for f in os.listdir(paths + '/'+ label)]
        new_extension = '.wav'
        f = paths+'/'+label
        pre, ext = os.path.splitext(f)
        os.rename(f, pre + "double" + new_extension)
#change_krz()


#
def process(file_path):

    audio_path =file_path
    samples, sample_rate = librosa.load(audio_path, sr=8000)  # Sample-Rate anpassen, falls nötig
    # Absolutwert der Amplitude berechnen
    amplitude = np.abs(samples)

    # Index der höchsten Amplitude finden
    max_index = np.argmax(amplitude)

    # Bereich um die maximale Amplitude (±200 ms)
    window_size = int(0.2 * sample_rate)  # 200 ms Fenster
    start = max(0, max_index - window_size)
    end = min(len(samples), max_index + window_size)
    # Bereich extrahieren
    loud_section = samples[start:end]
    #print(loud_section)
    file_length = len(samples) / sample_rate
    #print(len(samples))
    #print(file_length)
    target_length = 1
    if file_length < target_length:
        #print("*** Datei ist kürzer als 1 Sek")
        target_length = sample_rate
        padding = target_length - len(samples)
        samples = np.pad(samples, (0, int(padding)), mode='constant')  # Auffüllen mit Nullen
        start = 0
        end = target_length*sample_rate
    elif file_length > target_length:
        #print("*** Datei ist länger als 1 Sek ***")
        length_sound = end-start
        diff = target_length*sample_rate - length_sound
        start = start - diff/2
        end = start + target_length*sample_rate
        if start < 0:
            start = 0
            end = 1*sample_rate
        #print(start, end, (end-start)/sample_rate)
        if len(samples) < end:
            #print("*** Datei wird verlängert ***")
            padding = end - len(samples)
            samples = np.pad(samples, (0, int(padding)), mode='constant')  # Auffüllen mit Nullen
    file_length = len(samples) / sample_rate


    trimmed_samples = samples[int(start) : int(end)]
    output_path = file_path
    sf.write(output_path, trimmed_samples, 8000)

    # Check if everything went correctly
    audio_path = output_path
    samples, sample_rate = librosa.load(audio_path, sr=8000)  # Sample-Rate anpassen, falls nötig
     # Absolutwert der Amplitude berechnen
    amplitude = np.abs(samples)
    max_index = np.argmax(amplitude)
    #print("Höchste Amplitude in der neuen Datei" ,max_index)
    # Bereich um die maximale Amplitude (±200 ms)
    window_size = int(0.2 * sample_rate)  # 200 ms Fenster
    start = max(0, max_index - window_size)
    end = min(len(samples), max_index + window_size)
    # Bereich extrahieren
    loud_section = samples[start:end]
    #print(loud_section)
    file_length = len(samples) / sample_rate
    #print(len(samples))
    #print(file_length)
"""

for label in labels:
    print(label)
    paths = f"D:/Speech_recognition/files1/" + label
    if os.path.isdir(paths):
        datas=os.listdir(paths)
        for i in datas:
            if i.endswith(".wav"):
                process(paths + "/"+ i)
                samples, sample_rate = librosa.load(paths+'/'+i, sr=8000)  # Sample-Rate anpassen, falls nötig
                print(len(samples))
"""
def bereich(name,n=1):
    audio_path = f"D:/Speech_recognition/not_preprocessed_files/{wort}/{name}"
    print(audio_path)
    samples, sample_rate = librosa.load(audio_path, sr=8000)  # Sample-Rate anpassen, falls nötig
     # Absolutwert der Amplitude berechnen
    amplitude = np.abs(samples)
    #print(amplitude)
    #print(amplitude)
    # Index der höchsten Amplitude finden
    max_index = np.argmax(amplitude)
    #print("Höchste Amplitude" ,max_index)
# Bereich um die maximale Amplitude (±200 ms)
    window_size = int(0.2 * sample_rate)  # 200 ms Fenster
    start = max(0, max_index - window_size)
    end = min(len(samples), max_index + window_size)
    # Bereich extrahieren
    loud_section = samples[start:end]
    #print(loud_section)
    file_length = len(samples) / sample_rate
    #print(len(samples))
    #print(file_length)
    target_length = 1
    if file_length < target_length:
        print("*** Datei ist kürzer als 1 Sek")
        target_length = sample_rate
        padding = target_length - len(samples)
        samples = np.pad(samples, (0, int(padding)), mode='constant')  # Auffüllen mit Nullen
        start = 0
        end = target_length*sample_rate
    elif file_length > target_length:
        print("*** Datei ist länger als 1 Sek ***")
        length_sound = end-start
        diff = target_length*sample_rate - length_sound
        start = start - diff/2
        end = start + target_length*sample_rate
        if start < 0:
            start = 0
            end = 1*sample_rate
        print(start, end, (end-start)/sample_rate)
        if len(samples) < end:
            print("*** Datei wird verlängert ***")
            padding = end - len(samples)
            
            samples = np.pad(samples, (0, int(padding)), mode='constant')  # Auffüllen mit Nullen
    file_length = len(samples) / sample_rate

    print(start, end)
    trimmed_samples = samples[int(start) : int(end)]
    output_path = f"D:/Speech_recognition/not_preprocessed_files/check/trimmed_audio_neu{n}.wav"
    sf.write(output_path, trimmed_samples, sample_rate)

    # Check if everything went correctly
    audio_path = output_path
    samples, sample_rate = librosa.load(audio_path, sr=8000)  # Sample-Rate anpassen, falls nötig
     # Absolutwert der Amplitude berechnen
    amplitude = np.abs(samples)
    max_index = np.argmax(amplitude)
    print("Höchste Amplitude in der neuen Datei" ,max_index)
    # Bereich um die maximale Amplitude (±200 ms)
    window_size = int(0.2 * sample_rate)  # 200 ms Fenster
    start = max(0, max_index - window_size)
    end = min(len(samples), max_index + window_size)
    # Bereich extrahieren
    loud_section = samples[start:end]
    #print(loud_section)
    file_length = len(samples) / sample_rate
    #print(len(samples))
    print(file_length)

"""
n = 0
for name in labels:
    print(name)
    n += 1
    bereich(name,n)

"""