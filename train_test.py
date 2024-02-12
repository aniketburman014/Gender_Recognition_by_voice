import pandas as pd
import numpy as np
import os
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

label = {
    "male": 1,
    "female": 0
}
vector_length=128
def load_data():
    if not os.path.isdir("results"):
        os.mkdir("results")
    if os.path.isfile("results/features.npy") and os.path.isfile("results/labels.npy"):
        X = np.load("results/features.npy")
        y = np.load("results/labels.npy")
        return X, y
    df = pd.read_csv("dataset_final.csv")
    n_male_samples = len(df[df['gender'] == 'male'])
    n_female_samples = len(df[df['gender'] == 'female'])
    
    print("Total samples:", len(df))
    print("Total male samples:", n_male_samples)
    print("Total female samples:", n_female_samples)
    
    male_samples = df[df['gender'] == 'male'].sample(n=30000, random_state=1)
    female_samples = df[df['gender'] == 'female'].sample(n=30000, random_state=1)
    df_sampled = pd.concat([male_samples, female_samples])
    n_samples = len(df_sampled)
    
    
    X = np.zeros((n_samples, vector_length))
    y = np.zeros((n_samples, 1))
    
    for i, (filename, gender) in tqdm.tqdm(enumerate(zip(df_sampled['filename'], df_sampled['gender'])), "Loading data", total=n_samples):
        features = np.load(filename)
        X[i] = features
        y[i] = label[gender]
    np.save("results/features", X)
    np.save("results/labels", y)
    return X, y


def create_model():
    model = Sequential()
    model.add(Dense(256, input_shape=(vector_length,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    
    model.summary()
    return model


def split_data(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7,stratify=y)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=7,stratify=y_train)
    
    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test
    }

X,y=load_data()
data=split_data(X,y)
model=create_model()
tensorboard = TensorBoard(log_dir="logs")
early_stopping = EarlyStopping(mode="min", patience=5, restore_best_weights=True)

batch_size = 64
epochs = 100


model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size, validation_data=(data["X_valid"], data["y_valid"]),
          callbacks=[tensorboard, early_stopping])


model.save("results/model.h5")


print(f"Evaluating the model using {len(data['X_test'])} samples...")
loss, accuracy = model.evaluate(data["X_test"], data["y_test"], verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")
