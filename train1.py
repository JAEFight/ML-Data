import os
import librosa   #for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from scipy.io import wavfile #for audio processing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import random
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D 
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from prepreprocess import process
import warnings
warnings.filterwarnings("ignore")


path = 'D:/Speech_recognition_files/files1'
labels = os.listdir(path)
print(labels)

def get_raw_wave(samples, sample_rate):
    samples, sample_rate = librosa.load(path+'yes/0a7c2a8d_nohash_0.wav', sr = 8000)
    ipd.Audio(samples, rate=sample_rate)
    print(sample_rate)
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Raw wave of ' + '..D:/Speech_recognition/files/yes/0a7c2a8d_nohash_0.wav')
    ax1.set_xlabel('time')
    ax1.set_ylabel('Amplitude')
    # sample_rate/len(samples) gibt an wie lange die Datei geht... z.b. 1 Sekunde
    # np.linespace(start, end, Anzahl der Punkten)
    zeit = np.linspace(0, len(samples) / sample_rate, len(samples))
    ax1.plot(zeit, samples)
    plt.show()

def no_of_records(path, plot=False):
    # Number of records
    labels=os.listdir(path)
    for i in labels:
        if os.path.isdir(os.path.join(path, i)) is False:
            labels.remove(i)
    #find count of each label and plot bar graph
    no_of_recordings=[]
    for label in labels:
        # Ein loop mit dem man bedingungen machen kann
        waves = [f for f in os.listdir(path + '/'+ label) if f.endswith('.wav')]
        no_of_recordings.append(len(waves))
        
    #plot
    if plot:
        plt.figure(figsize=(30,5))
        index = np.arange(len(labels))
        plt.bar(index, no_of_recordings)
        plt.xlabel('Commands', fontsize=12)
        plt.ylabel('No of recordings', fontsize=12)
        plt.xticks(index, labels, fontsize=15, rotation=60)
        plt.title('No. of recordings for each command')
        plt.show()
    else:
        return labels
    

def preprocess(labels, path):
    all_label = []
    all_wave = []
    waves = {}
    print("*** übersicht über alle Label und alle Dateien wird erstellt *** ")
    for label in labels:
        #waves = [f for f in os.listdir(path + '/'+ label) if f.endswith('.wav')]
        temp_list = []
        is_dir=os.path.isdir(path + '/'+ label)
        if is_dir:
            for f in os.listdir(path + '/'+ label):
                if f.endswith('.wav'):
                    temp_list.append(f)
            waves[label] = temp_list
            print(label)
            # Waves ist ein Dict(hashmap) in welcher Listen sind, welche alle Dateien zu einem Label enthalten
            n = 0
            for wav in waves[label]: # Also quasi in temp_list
                
                samples, sample_rate = librosa.load(path + '/' + label + '/' + wav, sr = 8000)
                #sample_rate, samples = wavfile.read(path + '/' + label + '/' + wav)
                duration = float(len(samples)/sample_rate)
                if duration != 1.0:
                    print(duration)
                    process(path + '/' + label + '/' + wav)
                    samples, sample_rate = librosa.load(path + '/' + label + '/' + wav, sr = 8000)
                    duration = float(len(samples)/sample_rate)
                    if duration == 1.0 and len(samples) == 8000:
                        print("Plus 1")
                        n += 1
                        all_wave.append(samples)
                        all_label.append(label)
                    else:
                        waves[label].remove(wav)
                else:
                #samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=8000)
                    n += 1
                    if(len(samples)== 8000) : 
                        all_wave.append(samples)
                        all_label.append(label)
            #print(f"*** Erfolgreiches resamblen in 8000er sample rate [{label}][{n}]***")
            #print(f"*** alle gültigen waves zu all_wave hinzugefügt [{label}][{n}]***")

    encoder = LabelEncoder()
    #print(all_label)
    encoded_labels=encoder.fit_transform(all_label) # Gives every label a numeric label [1,2,3]
    #print(encoded_labels)
    print("*** Labels to numeric labels transformed ***")
    classes= list(encoder.classes_) # Let's me transform the numeric labels back into classes :   original_label = classes[numeric_label]
    print("*** classes to transform numeric labels back to original ***")
    y=keras.utils.to_categorical(encoded_labels, num_classes=len(labels))
    
    print("*** labels to categorical labels (mandatory for conv1d) ***")
    """""
    After one-hot encoding (len(labels) = 3):
    y = [
    [1, 0, 0],  # Class 0
    [0, 1, 0],  # Class 1
    [0, 0, 1],  # Class 2
    [0, 1, 0]   # Class 1
    ]
    """
    all_wave = np.array(all_wave).reshape(-1,8000,1)
    print("*** succesfully reshaped all waves ***")
    #decoded_labels = [encoder.classes_[i] for i in encoded_labels]
    #print(len(y), len(all_wave)) 
    #print("Decoded Labels:", decoded_labels)
    if len(y) != len(all_wave):
        list1 = y.tolist()
        list1.pop()
        y = np.asarray(list1)
    
    #print("Anzahl der Samples in all_wave:", len(all_wave))
    #print("Anzahl der Labels in y:", len(y))
    print("Numerical arrays",encoded_labels)
    print("y nach categorical(biite 3d array numeric)",y)
    #x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2 ,random_state=777,shuffle=True)
    x_tr, x_val, y_tr, y_val = train_test_split(all_wave, y,stratify=encoded_labels, test_size=0.2,random_state=777,shuffle=True)
    print(y_val)
    print(np.argmax(y_val[1]))
    print(classes)
    #correct_answer = classes[np.argmax(y_val[index])]
    return x_tr, x_val, y_tr, y_val, classes, all_wave, y, encoded_labels

def preprocess_compromized(labels, path):
    all_label = []
    all_wave = []
    waves = {}
    print("*** Kompromierte übersicht über alle Label und alle Dateien wird erstellt *** ")
    for label in labels:
        #waves = [f for f in os.listdir(path + '/'+ label) if f.endswith('.wav')]
        temp_list = []
        for f in os.listdir(path + '/'+ label):
            if f.endswith('.wav'):
                temp_list.append(f)
        waves[label] = temp_list
        print(label)
        # Waves ist ein Dict(hashmap) in welcher Listen sind, welche alle Dateien zu einem Label enthalten
        n = 0
        for wav in waves[label]: # Also quasi in temp_list
            if n >= 150:
                break
            samples, sample_rate = librosa.load(path + '/' + label + '/' + wav, sr = 8000)
            #sample_rate, samples = wavfile.read(path + '/' + label + '/' + wav)
            duration = float(len(samples)/sample_rate)
            if duration != 1.0:
                #print(duration)
                waves[label].remove(wav)
            else:
            #samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=8000)
                n += 1
                if(len(samples)== 8000) : 
                    all_wave.append(samples)
                    all_label.append(label)
    encoder = LabelEncoder()
    encoded_labels=encoder.fit_transform(all_label) # Gives every label a numeric label [1,2,3]
    print("*** Labels to numeric labels transformed ***")
    classes= list(encoder.classes_) # Let's me transform the numeric labels back into classes :   original_label = classes[numeric_label]
    print("*** classes to transform numeric labels back to original ***")
    y=keras.utils.to_categorical(encoded_labels, num_classes=len(labels))
    
    print("*** labels to categorical labels (mandatory for conv1d) ***")
    """""
    After one-hot encoding (len(labels) = 3):
    y = [
    [1, 0, 0],  # Class 0
    [0, 1, 0],  # Class 1
    [0, 0, 1],  # Class 2
    [0, 1, 0]   # Class 1
    ]
    """
    all_wave = np.array(all_wave).reshape(-1,8000,1)
    print("*** succesfully reshaped all waves ***")

    if len(y) != len(all_wave):
        list1 = y.tolist()
        list1.pop()
        y = np.asarray(list1)
    
    print("Numerical arrays",encoded_labels)
    print("y nach categorical(biite 3d array numeric)",y)
    x_tr, x_val, y_tr, y_val = train_test_split(all_wave, y,stratify=encoded_labels, test_size=0.2,random_state=777,shuffle=True)
    return x_tr, x_val, y_tr, y_val, classes, all_wave, y, encoded_labels

def train(x_tr, x_val, y_tr, y_val, labels):
    file1 = open("D:/Speech_recognition/word_structure.txt", "w")
    for i in labels:
        file1.write(i)
        file1.write('\n')
    file1.close()
    y_tr = np.array(y_tr)
    print(f"x_tr shape: {x_tr.shape}, dtype: {x_tr.dtype}")
    print(f"y_tr shape: {y_tr.shape}, dtype: {y_tr.dtype}")
    K.clear_session()

    inputs = Input(shape=(8000,1)) #Eine Audiodatei mit 8000 Zeitpunkten und 1 Kanal

    #First Conv1D layer
    conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs) # Filteranzahl = 8, 8verschieden Merkmale. Kernelgröße = 13 benachbarte Zeitpunkte pro Filter
    conv = MaxPooling1D(3)(conv) #Reduziert die Ausgabe um den Faktor 3, um die Dimensionen zu verringern und wichtige Merkmale hervorzuheben
    conv = Dropout(0.3)(conv) #Setzt zufällig 30% der Neuronen auf 0, um Overfitting zu verhindern

    #Second Conv1D layer, zunehmend komplexere Merkmale und Fokus auf kleinere Muster
    """
    Zunehmende Filteranzahl: 16, 32, 64 → Lernen zunehmend komplexere Merkmale.
    Abnehmende Kernelgröße: 11, 9, 7 → Fokus wird enger auf kleinere Muster gelegt.

    """
    conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    #Third Conv1D layer 
    conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    #Fourth Conv1D layer
    conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    #Flatten layer
    conv = Flatten()(conv) # Umwnadlung mehrdimensionaler Ausgabe in einen 1D-Vektor um

    #Dense Layer 1
    conv = Dense(256, activation='relu')(conv) #256 Neuronen
    conv = Dropout(0.3)(conv)

    #Dense Layer 2, weiterer Kompromierungsschritt
    conv = Dense(128, activation='relu')(conv)
    conv = Dropout(0.3)(conv)

    outputs = Dense(len(labels), activation='softmax')(conv)
    """
    Anzahl der Neuronen: len(labels) (eine Ausgabe für jede Klasse).
    Softmax-Aktivierung: Gibt Wahrscheinlichkeiten für jede Klasse zurück.
    """
    #Baut das Modell zusammen und zeigt die Architektur an.
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 
    mc = ModelCheckpoint('Final.keras', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model.summary()
    print("*** training the model ***")
    history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es,mc], batch_size=56, validation_data=(x_val,y_val))
    print("*** Visualizing the performance ***")
    plt.plot(history.history['loss'], label='train') 
    plt.plot(history.history['val_loss'], label='test') 
    plt.legend()
    plt.show()

    # Save the model:
    model.save('d:/Speech_recognition/model1/Final.keras')
    model.summary()

def train_with_model(x_tr, x_val, y_tr, y_val, existing_model_path):
    # Laden des alten Modells
    model = existing_model_path
    
    # Die Eingabeschicht beibehalten
    old_model_input = model.input
    
    # Neue Ausgabeschicht für 18 Klassen (anstatt 17)
    new_output = Dense(18, activation='softmax', name='dense_new')(model.layers[-2].output)  # -2 ist die penultimate Schicht

    # Erstelle das neue Modell
    new_model = Model(inputs=old_model_input, outputs=new_output)

    # Kompiliere das Modell
    new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Zeige die Architektur des neuen Modells an
    new_model.summary()

    # Training mit den neuen Daten
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
    mc = ModelCheckpoint('new_all_words.keras', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    print("*** Training das Modell ***")
    history = new_model.fit(x_tr, y_tr, epochs=100, callbacks=[es, mc], batch_size=32, validation_data=(x_val, y_val))

    """
        print("*** Visualizing the performance ***")
        plt.plot(history.history['loss'], label='train') 
        plt.plot(history.history['val_loss'], label='test') 
        plt.legend()
    """
    # Save das Modell
    new_model.save('d:/Speech_recognition/model1/new_all_words.keras')
    new_model.summary()

def retrain(model):
    splits = preprocess(labels,path)
    all_wave = splits[5]
    y = splits[6]
    encoded_labels = splits[7]

    x_train, x_val, y_train, y_val = train_test_split(all_wave, y, stratify=encoded_labels,test_size=0.1, random_state=42, shuffle=True)
    # Weitere Trainingszeit für das Modell
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)  # Patience = nach wievielen "nicht-verbesserungen" wird early gestoppt, min-delta legt fest ab was für einer Verbesserung in der accuraccy keine Verbessetung mehr fest gestellt wird
    mc = ModelCheckpoint('Final.keras', monitor='val_acc', verbose=1, save_best_only=True, mode='max') # Die neue Datei wird so gespeichert falls das Programm zwischendrin abbricht
    history = model.fit(x_train, y_train, epochs=80, batch_size=200, validation_data=(x_val, y_val),callbacks=[es,mc])
    model.save('Final.keras')
    model.summary()
    # Theoretisch könnte man noch Adam optimieren also die Learningrate ernidrigen wenn man die Batch size verringert
loaded_model = load_model("d:/Speech_recognition/model1/all_words.keras")
classes =  []
def test():
    print("*** Jetzt wird getestet ***")
    
    file1 = open("D:/Speech_recognition/word_structure.txt", "r")
    for class1 in file1:
        classes.append(class1.removesuffix('\n'))
    file1.close()
    print(classes)
    # classes = ['background', 'eight', 'five', 'four', 'happy', 'marvin', 'nine', 'off', 'on', 'one', 'seven', 'six', 'stop', 'three', 'two','weather', 'zero'] # Muss die gleiche Reihenfolge haben, wie die Ordnerstruktur
    def predict(audio):
        prob=loaded_model.predict(audio.reshape(1,8000,1))
        print(prob)
        index=np.argmax(prob[0])
        print(index)
        return classes[index]
    specific_labels = ['marvin', 'weather', 'yes', 'no', 'stop'] # Falls man nur spezifische Labels testen möchte
    #specific_indices = [i for i, label in enumerate(np.argmax(y_val, axis=1)) if classes[label] in specific_labels]  #theoretisch aber praktisch eher zu aufwändig, eigtl sollte man bei specific labels die eingeben, welche man testen soll und hier wird dann eine Liste mit den zugehörigen Idices erstellt
    specific_indices = [1,2,3,4,5]

    # Statistics
    counter_correct = 0
    counter_false = 0

    falses = []
    for index in range(0,1000):
        index = random.randint(0, len(classes)-1)
        print(index)
        # Random File
        print(path + '/'+ classes[index])
        file_directory = os.listdir(path + '/'+ classes[index])
        file_index = random.randint(1, len(file_directory)-1)
        print(file_index, len(file_directory))
        audio_file = file_directory[file_index]
        print(audio_file)
        samples, sample_rate = librosa.load(path + '/' + classes[index] + '/' + audio_file, sr = 8000)
        print(len(samples))
        if len(samples) < 8000:
            print(audio_file, "Hat die falsche Anzahl an Samples")
            pass
        else:
            #samples = x_val[index].ravel()  # Extrahiert das Audiosample als flachen Vektor
            correct_answer = classes[index]
            print("Audio(Korrekt):", correct_answer)  # Gibt die wahre Klasse aus (z. B. 'weather', 'marvin', etc.)
            prediction = predict(samples)
            print("Text(Predicted):", prediction)  # Gibt das vorhergesagte Label des Modells aus
            if correct_answer == prediction:
                counter_correct += 1
            else:
                counter_false += 1
                s = "Audio:"+ correct_answer + "Prediction:" + prediction
                falses.append(s)
            print(correct_answer, prediction)
    return counter_correct, counter_false, falses

classes = []
comm = input("Test(t), Training(T), Retraining(r), Nichts(N):, Train with existing model(tm)")
if comm == 't':
    result = test()
    print(result)
elif comm == 'T':
    splits = preprocess(labels,path)
    x_tr = splits[0]
    x_val = splits[1]
    y_tr = splits[2]
    y_val = splits[3]
    classes = splits[4]
    all_wave = splits[5]
    y = splits[6]
    encoded_labels = splits[7]
    train(x_tr, x_val, y_tr, y_val,labels)
elif comm == 'r':
    retrain(loaded_model)
elif comm == 'tm':
    splits = preprocess(labels,path)
    x_tr = splits[0]
    x_val = splits[1]
    y_tr = splits[2]
    y_val = splits[3]
    classes = splits[4]
    all_wave = splits[5]
    y = splits[6]
    encoded_labels = splits[7]
    train_with_model(x_tr, x_val, y_tr, y_val,loaded_model)
else:
    print("*** Funktionen stehen zur Vergügung ***")
def test_realtime(file_path):
    classes = []
    file1 = open("D:/Speech_recognition/word_structure.txt", "r")
    for class1 in file1:
        classes.append(class1.removesuffix('\n'))
    file1.close()
    print(classes)
    def predict(audio):
        prob=loaded_model.predict(audio.reshape(1,8000,1))
        index=np.argmax(prob[0])
        return classes[index]
    samples, sample_rate = librosa.load(file_path, sr = 8000)
    prediction = predict(samples)
    return prediction
