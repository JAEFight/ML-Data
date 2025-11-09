import pandas as pd
import shutil

data = pd.read_csv('D:/Speech_recognition/dataset/metadata.csv')

lights = []
music = []
volume1 = []
heat = []
df = pd.DataFrame(data[["path", "speaker", "action", "object", "current_language", "gender", "age_range"]])
args = ["lights", "music", "volume", "heat"]

for obj in args:
    new_df = df.loc[df['object'] == f"{obj}"]
    destination_path = f"D:/Speech_recognition/dataset/{obj}"
    #print(new_df["path"])
    print(obj)
    for index, row in new_df.iterrows():
        #print(f"Index: {index}, Value: {row['path']}")
        source_path = f"D:/Speech_recognition/{row['path']}"
        shutil.copy(source_path, destination_path)
        #print("Datei erfolgreich kopiert!")
