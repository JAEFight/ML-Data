import os
paths = "D:/Speech_recognition/en/clips/"
labels=os.listdir(paths)
#find count of each label and plot bar graph

for label in labels:
    # Ein loop mit dem man bedingungen machen kann
    waves = [f for f in os.listdir(paths + '/'+ label) if f.endswith('.opus')]

    new_extension = '.wav'
    for f in waves:
        f = paths+'/'+label+'/'+f
        if f.endswith('.opus'):
            pre, ext = os.path.splitext(f)
            os.rename(f, pre + new_extension)
