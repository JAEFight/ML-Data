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
import random
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
#from keras import utils as np_utils 
import warnings
warnings.filterwarnings("ignore")


path = 'D:/Speech_recognition/files/'
# Resamble by changing the sample rate
samples, sample_rate = librosa.load(path+'yes/0a7c2a8d_nohash_0.wav', sr = 8000)
def get_raw_wave(samples, sample_rate):
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
labels = no_of_records(path)
print(labels)
def get_duration(labels, path):
    waves = {}
    averages = {}
    print("*** übersicht über alle Label und alle Dateien wird erstellt *** ")
    for  label in labels:
        #waves = [f for f in os.listdir(path + '/'+ label) if f.endswith('.wav')]
        temp_list = []
        for f in os.listdir(path + '/'+ label):
            if f.endswith('.wav'):
                temp_list.append(f)
        waves[label] = temp_list
        # Waves ist ein Dict(hashmap) in welcher Listen sind, welche alle Dateien zu einem Label enthalten
        sum_duration = 0
        duration_of_recordings=[]
        for wav in waves[label]:
            sample_rate, samples = wavfile.read(path + '/' + label + '/' + wav)
            duration = float(len(samples)/sample_rate)
            sum_duration += duration
            duration_of_recordings.append(duration)
            if duration != 1:
                waves[label].remove(wav)
            #print(len(duration_of_recordings))
            #print(wav)
        print(f"*** Durations are documented [{label}]***")
        avg_duration = sum_duration/len(duration_of_recordings)
        averages[label] = avg_duration
        print("*** Averages erstellt ***")
        summary = label, sum_duration, len(duration_of_recordings), avg_duration
        return summary
#get_duration(labels,path)

def preprocess(labels, path):
    all_label = []
    all_wave = []
    waves = {}
    print("*** übersicht über alle Label und alle Dateien wird erstellt *** ")
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
        #print(f"*** Erfolgreiches resamblen in 8000er sample rate [{label}][{n}]***")
        #print(f"*** alle gültigen waves zu all_wave hinzugefügt [{label}][{n}]***")

    print(len(waves))
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
if input("Test? :") != "t":
    splits = preprocess(labels,path)
else:
    splits = preprocess_compromized(labels,path)

"""
print("x_tr",splits[0])
print("x_val",splits[1])
print("y_tr",splits[2])
print("y_val",splits[3])
"""
print("y_tr",splits[2])
def train(x_tr, x_val, y_tr, y_val, labels):
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
    mc = ModelCheckpoint('model_weather.hdf5.keras', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model.summary()
    print("*** training the model ***")
    history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))
    print("*** Visualizing the performance ***")
    plt.plot(history.history['loss'], label='train') 
    plt.plot(history.history['val_loss'], label='test') 
    plt.legend()
    #plt.show()

    # Save the model:
    model.save('model_weather.keras')
x_tr = splits[0]
x_val = splits[1]
y_tr = splits[2]
y_val = splits[3]
classes = splits[4]
all_wave = splits[5]
y = splits[6]
encoded_labels = splits[7]
#train(x_tr, x_val, y_tr, y_val,labels)

def retrain(all_wave,y, encoded_labels, model):
    x_train, x_val, y_train, y_val = train_test_split(all_wave, y, stratify=encoded_labels,test_size=0.1, random_state=42, shuffle=True)
    # Weitere Trainingszeit für das Modell
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 
    mc = ModelCheckpoint('best_model.hdf5.keras', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    history = model.fit(x_train, y_train, epochs=80, batch_size=42, validation_data=(x_val, y_val),callbacks=[es,mc])
    model.save('updated_model.keras')
    model.summary()


loaded_model = load_model("model_weather.keras")
loaded_model.summary()
def test():
    def predict(audio):
        prob=loaded_model.predict(audio.reshape(1,8000,1))
        #print(prob)
        index=np.argmax(prob[0])
        #print(index)
        return classes[index]
    counter_correct = 0
    counter_false = 0
    falses = []
    for i in range(0,150):
        index = random.randint(0, len(x_val) - 1)  # Wählt zufälligen Index aus Validierungsdaten
        samples = x_val[index].ravel()  # Extrahiert das Audiosample als flachen Vektor
        #print(classes)
        correct_answer = classes[np.argmax(y_val[index])]
        #print("Audio:", correct_answer)  # Gibt die wahre Klasse aus (z. B. 'yes', 'no', etc.)
        prediction = predict(samples)
        #print("Text:", prediction)  # Gibt das vorhergesagte Label des Modells aus
        
        if correct_answer == prediction:
            counter_correct += 1
        else:
            counter_false += 1
            s = "Audio:"+ correct_answer + "Prediction:" + prediction
            falses.append(s)

    return counter_correct, counter_false, falses

#retrain(all_wave,y, encoded_labels, loaded_model)

#Test with user made voice record
def own_test():
    def predict(audio):
        prob=loaded_model.predict(audio.reshape(1,8000,1))
        print(prob)
        index=np.argmax(prob[0])
        print(index)
        return classes[index]
    samplerate = 8000  
    duration = 1 # seconds
    path = 'D:/Speech_recognition/'
    samples, sample_rate = librosa.load(path+'/weather_test.wav', sr = samplerate, duration = duration)
    print(len(samples))
    if len(samples) <= 8000:
        samples = np.pad(samples, (0, 8000 - len(samples)), mode='constant')
        print("own test: ")
        prediction = predict(samples)
        print(prediction)
    return prediction
own_test()
#print("my_best_model.keras,  Correct/False", test())
"""
first_val = test()
loaded_model = load_model("my_best_model.keras")
print("my_best_model.keras,  Correct/False", test())
print("updated_model.keras,  Correct/False", first_val)
"""
#t1 = own_test()
#loaded_model = load_model("my_best_model.keras")
#t2 = own_test()
#print(f"My_best_model : {t2}, Updated_model : {t1}")