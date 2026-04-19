import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("Booting up the ISL Training Module")

DATA_PATH = 'Custom_ISL_Data'
actions = np.array(['Alright', 'Good Morning', 'Good afternoon', 'Good evening', 'Good night', 'Hello', 'How are you', 'Pleased', 'Thank you'])

label_map = {label:num for num, label in enumerate(actions)}

print(f"Loading dataset from '{DATA_PATH}'")
sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        continue
        
    for file in os.listdir(action_path):
        if file.endswith('.npy'):
            res = np.load(os.path.join(action_path, file))
            sequences.append(res)
            labels.append(label_map[action])

print(f"Loaded {len(sequences)} total video sequences.")

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples.")

print("Building the Neural Network Architecture...")
model = Sequential()

model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(40, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Starting Training Process...")

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test),
    epochs=100, 
    callbacks=[early_stop]
)

MODEL_NAME = 'isl_custom_model.h5'
model.save(MODEL_NAME)
print(f"\nSUCCESS! Model fully trained and saved as '{MODEL_NAME}'")
print("You are ready to plug this into your real-time translator app!")