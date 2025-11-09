from tensorflow import keras
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()
labels = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 
'wow', 'yes', 'zero', '_background_noise_']

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
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.summary()