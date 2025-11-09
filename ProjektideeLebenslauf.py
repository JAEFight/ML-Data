# Website mit Flask um einen Lebenslauf oder andere Dokumente zu erstellen
import os
path = 'D:/Speech_recognition/files1/'
labels = os.listdir(path)
file1 = open("D:/Speech_recognition/word_structure.txt", "w")

for i in labels:
    file1.write(i+ "\n")
file1.close()
classes =  []
file1 = open("D:/Speech_recognition/word_structure.txt", "r")
for class1 in file1:
    classes.append(class1.removesuffix('\n'))
file1.close()
