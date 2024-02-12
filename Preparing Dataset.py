#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glob
import os
import pandas as pd
import numpy as np
import shutil
import librosa
from tqdm import tqdm


# In[3]:


def extract_feature(file_name,**kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result
    


# In[4]:


import os
import glob

dirname = "Downloads/archive"
if not os.path.isdir(dirname):
    os.mkdir(dirname)
    

csv_files = glob.glob(dirname + "/*.csv") 
for j, csv_file in enumerate(csv_files):
    print("[+] Processing", csv_file)
    df=pd.read_csv(csv_file)
    df_new=df[["filename","gender"]]
    print("Before Removing NAN , Other Values:", len(df_new), "rows")
    df_new = df_new[df_new['gender'].isin(['male', 'female'])]
    print("After Removing:", len(df_new), "rows")
    new_csv_file = os.path.join( csv_file)
    df_new.to_csv(new_csv_file, index=False)
    folder_name,_=csv_file.split(".")
#     print(folder_name)
    audio_files = glob.glob(f"{folder_name}/*")
#     print(audio_files)
    
    audio_filenames = set(df_new["filename"])
    
    for i,audio_file in tqdm(list(enumerate(audio_files)),f"Extracting features of {folder_name}"):
        splited = os.path.split(audio_file)
#         print(splited)
        audio_filename=f"{os.path.split(splited[0])[-1]}/{splited[-1]}"
#         print(audio_filename)
        if audio_filename in audio_filenames:
            src_path = f"Downloads/archive/{audio_filename}"
            target_path = f"Downloads/Data/{audio_filename}"
#             print(src_path)
#             print(target_path)
            if not os.path.isdir(os.path.dirname(target_path)):
                os.makedirs(os.path.dirname(target_path))
            features = extract_feature(src_path, mel=True)
            target_filename = target_path.split(".")[0]
#             print(target_filename)
            np.save(target_filename, features)


# In[5]:


dirname = "Downloads/archive"
if not os.path.isdir(dirname):
    os.mkdir(dirname)
    
all_data_df = pd.DataFrame()
csv_files = glob.glob(dirname + "/*.csv") 
for j, csv_file in enumerate(csv_files):
    df=pd.read_csv(csv_file)
    for index, row in df.iterrows():
        
        spli = row['filename'].split('.')
        path=path = f"Downloads/Data/{spli[0]}.npy"
        if os.path.exists(path):
            all_data_df = all_data_df.append({'filename': f"Data/{spli[0]}.npy", 'gender': row['gender']}, ignore_index=True)
            
            
            


# In[6]:


all_data_df.to_csv('output.csv', index=False)


# In[7]:


n_samples = len(all_data_df)
# get total male samples
n_male_samples = len(all_data_df[all_data_df['gender'] == 'male'])
# get total female samples
n_female_samples = len(all_data_df[all_data_df['gender'] == 'female'])
print("Total samples:", n_samples)
print("Total male samples:", n_male_samples)
print("Total female samples:", n_female_samples)

