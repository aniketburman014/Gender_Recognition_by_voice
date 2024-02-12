import os 
import wave
import time
import threading
import tkinter as tk
import pyaudio 
import datetime
from tkinter import filedialog
from array import array
import librosa
import numpy as np
from sys import byteorder
from struct import pack
from keras.models import load_model

class VoiceRecorder:
    THRESHOLD = 500
    MAXIMUM = 16384

    def __init__(self):
        self.root = tk.Tk()
        self.root.resizable(False,False)
        
        
        
        self.button = tk.Button(text="Record 🎙️",font=("Arial",60,"bold"),command=self.click_handler)
        self.button.pack()
        self.label = tk.Label(text="00:00:00")
        self.label.pack()
        self.recording = False
        self.file_path=None
        self.root.mainloop()
        

    def is_silent(self, snd_data):
        
        return max(snd_data) < self.THRESHOLD



    def normalize(self, snd_data):
        times = float(self.MAXIMUM)/max(abs(i) for i in snd_data)
        r = array('h')
        for i in snd_data:
            r.append(int(i*times))
        return r

    def trim(self, snd_data):
        def _trim(snd_data):
            snd_started = False
            r = array('h')
            for i in snd_data:
                if not snd_started and abs(i)>500:
                    snd_started = True
                    r.append(i)
                elif snd_started:
                    r.append(i)
            return r

        # Trim to the left
        snd_data = _trim(snd_data)

        # Trim to the right
        snd_data.reverse()
        snd_data = _trim(snd_data)
        snd_data.reverse()
        return snd_data

    def add_silence(self, snd_data, seconds=0.25):
        r = array('h', [0 for i in range(int(seconds*44100))])
        r.extend(snd_data)
        r.extend([0 for i in range(int(seconds*44100))])
        return r

    def click_handler(self):
        if self.recording:
            self.recording = False
            self.button.config(fg="black")
        else:
            self.recording = True
            self.button.config(fg="red")
            threading.Thread(target=self.record).start()

    def record(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, rate=44100, channels=1, input=True, frames_per_buffer=1024)
        frames = array('h')
        start = time.time()
        while self.recording:
            data = stream.read(1024)
            frames.extend(array('h', data))
            passed = time.time() - start
            secs = passed % 60
            mins = passed // 60
            hours = mins // 60
            if self.root:
                self.label.config(text= f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        frames = self.normalize(frames)
        frames = self.trim(frames)
        frames = self.add_silence(frames, 0.5)

        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        file = f"test_samples/{date_time}.wav"
        sound_file = wave.open(file,"wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(frames.tobytes())
        sound_file.close()
        self.file_path = file
        self.root.destroy()


class AudioUploader:
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.resizable(False,False)
        self.heading = tk.Label( text="Voice Gender Detection", font=("Arial", 50))
        self.heading.pack(pady=30)
        
        self.upload_button = tk.Button(self.root,font=("Arial",50,"bold"), text="Upload Audio 📥", command=self.upload)
        self.upload_button.pack(pady=20)
        
        self.record_button = tk.Button(self.root,font=("Arial",50,"bold"), text="Record Audio 🎙️", command=self.record)
        self.record_button.pack()
        self.root.geometry('800x600')
        
        self.file_path=None
        self.root.mainloop()
        
    def upload(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3")])
        if self.file_path:
            self.root.destroy()
    def record(self):
        self.root.destroy()
        k=VoiceRecorder()
        self.file_path=k.file_path






def extract_feature(file_name, **kwargs):
    
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

    
    
if __name__ == "__main__":
    
     
    model=load_model("results/model.h5")
    
    file=AudioUploader()
    
    features = extract_feature(file.file_path, mel=True).reshape(1, -1)
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "Male" if male_prob > female_prob else "Female"
    
    
    result = f''' {gender}
    
    Probabilities:     Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%'''
    print("Result:", gender)
    print(f"Probabilities:     Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")
    
    root = tk.Tk()
    root.geometry('800x600')
    root.resizable(False, False)

    heading = tk.Label(root, text="Voice Gender Detection", font=("Arial", 50),fg='#f00')
    heading.pack(pady=30)


    result_label = tk.Label(root, text=result, font=("Arial", 20,))
    result_label.pack()

    root.mainloop()
    
    
    
    
    
    
   
    
    
    
        
        
