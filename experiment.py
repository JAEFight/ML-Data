import pyaudio
import numpy as np
import wave
from train1 import test_realtime
from prepreprocess import process
from get_weather import weather_values
from timer import start_timer, stop_timer
import time
from play_song import happy_song
# Audio-Parameter
FORMAT = pyaudio.paInt16  # 16-Bit PCM
CHANNELS = 1              # Mono
RATE = 8000               # Abtastrate: 8 kHz
CHUNK = 1024              # Frames pro Buffer
THRESHOLD = 1000  # Schwelle für die Amplitude
SILENCE_DURATION = 6  # Sek. Stille bevor Aufnahme endet

def is_speech(data, threshold):
    """Prüft, ob Daten über der Amplitudenschwelle liegen."""
    return np.abs(np.frombuffer(data, dtype=np.int16)).max() > threshold

def record_command(output_filename):
    # PyAudio-Objekt erstellen
    p = pyaudio.PyAudio()
    # Mikrofon-Stream öffnen
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("Aufnahme gestartet (Drücke STRG+C, um zu stoppen)")
    silent_chunks = 0
    recording = True
    frames = []
    while recording:

        data = stream.read(CHUNK)
        frames.append(data)
        # Hier miuss snochmal ran
        if is_speech(data, THRESHOLD): # Mikrofon schlägt aus
            silent_chunks = 0
            #print("Sound erkannt")
            print("+")
        else:
            silent_chunks += 1
            silence_in_seconds = silent_chunks / (RATE /CHUNK)
            print("-")
            if silence_in_seconds > 1.2:
                print("Aufnahme beendet.")
                recording = False

    # Stream und PyAudio schließen
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Speicher in einer Wav datei
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio wurde gespeichert als '{output_filename}'.")

    """for i in range(1,2):
        print(i)
        if i == 1:
            OUTPUT_FILENAME = "D:/Speech_recognition/realtime_audio/activate_com.wav"
            record(OUTPUT_FILENAME)
            process(OUTPUT_FILENAME)
            ergebnis = test_realtime(OUTPUT_FILENAME)
            print(ergebnis)
            if ergebnis == 'marvin':
                print("Wurde Aktiviert")
                OUTPUT_FILENAME = "D:/Speech_recognition/realtime_audio/action_com.wav"
                record(OUTPUT_FILENAME)
                process(OUTPUT_FILENAME)
                
                print(test_realtime(OUTPUT_FILENAME))
                print(weather_values("Wiesbaden",1)[1])
        #record(OUTPUT_FILENAME)"""


def record(output_filename):
    # PyAudio-Objekt erstellen
    p = pyaudio.PyAudio()
    # Mikrofon-Stream öffnen
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("Aufnahme gestartet (Drücke STRG+C, um zu stoppen)")
    silent_chunks = 0
    recording = True
    frames = []
    first_sound = False
    while recording:

        data = stream.read(CHUNK)
        frames.append(data)
        # Hier miuss snochmal ran
        if is_speech(data, THRESHOLD):#Mikrofon schlägt aus
            silent_chunks = 0
            #print("Sound erkannt")
            print("+")
            first_sound = True
            silent_chunks = 0
        else:
            silent_chunks += 1
            silence_in_seconds = silent_chunks / (RATE /CHUNK)
            print("-")

        if first_sound:
            silent_chunks += 1
            print("-")
            silence_in_seconds = silent_chunks / (RATE /CHUNK)
            if silence_in_seconds > 1.2:
                recording = False


    # Stream und PyAudio schließen
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Speicher in einer Wav datei
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio wurde gespeichert als '{output_filename}'.")


print("Zuhörmodus Aktiv")
z = 0 # Um zu checken, wie oft Marvin NICHT gesagt wird, damit das listen nicht mehr aktiviert ist.
timer_started = False
listen = True
playing = False
while listen:
    print("Zuhörmodus Aktiv")
    # Zuhörmodus
    activation_FILENAME = "D:/Speech_recognition/realtime_audio/activate_com.wav"
    record(activation_FILENAME)
    process(activation_FILENAME)
    ergebnis = test_realtime(activation_FILENAME)
    print("Eingabe : " + ergebnis)
    
    if ergebnis == "marvin":
        print("Wurde Aktiviert --> Lampe leuchtet --> 0.5 sek delay")
        #time.sleep(0.5)
        active = True
        c = 0 # Um zu checken, wie oft kein Command erkannt wird, damit das listen nicht mehr aktiviert ist.
        
        while active:
            command_FILENAME = "D:/Speech_recognition/realtime_audio/action_com.wav"
            record_command(command_FILENAME)
            process(command_FILENAME)
            command = test_realtime(command_FILENAME)
            print(command)
            if command != "background" and command != "off":  # Ich brauche noch viele leere Background dateien
                if command == "weather":
                    print(weather_values("Wiesbaden",1)[1])
                    active = False

                elif command == "go":# Go ---> bis stop und zeit wird gestoppt
                    start = start_timer()
                    timer_started = True
                    active = False

                elif command == "stop":
                    if timer_started:
                        t = stop_timer(start)
                        print("Verstrichene Zeit : " + str(t) + "Sekunden")
                    active = False
                elif command == "happy":
                    happy_song()
                    playing = True
                    ative = False
                    # on or off für eine kleine Lampe 
            else: # Zeit die verstreichen darf, wenn aktiv aber kein Command
                c = c + 1
                if c == 3:
                    active = False
                
    elif ergebnis == "stop":
        listen = False
    else:
        print("Kein Marvin, keine Activation")

### Testen ob ich bei der endlosen Abfrage am Anfang auch direkt marvin erkennen kann, bzw die   Activator Aufahme von Anfang an starten lassen

#### Auch noch das Wort stop trainieren

#Beim Training die Reihenfolge der Worte in einer externen Datei abspeichern auf welche, dann auch zugegriffen wird beim testen