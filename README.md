# AI-Sign-Language-Detection-ISL-

A real-time **Indian Sign Language (ISL)** recognition system that uses a webcam, hand keypoint detection, and a deep learning model to translate ISL gestures into text — live.

---

## Supported Signs

`Alright` · `Good Morning` · `Good Afternoon` · `Good Evening` · `Good Night`
`Hello` · `How Are You` · `Pleased` · `Thank You`

---

## How It Works

1. **Data Collection** (`collect_data.py`) — Records hand keypoint sequences (via MediaPipe) for each sign and saves them as `.npy` files.
2. **Model Training** (`train_model.py`) — Trains a 3-layer LSTM neural network on the collected sequences.
3. **Real-Time Translation** (`app.py`) — Uses the trained model to predict ISL gestures live from a webcam feed.

---

## Tech Stack

- **Python** — Core language
- **TensorFlow / Keras** — LSTM model training and inference
- **MediaPipe** — Hand landmark detection (21 keypoints × 2 hands)
- **OpenCV** — Webcam capture and frame rendering
- **NumPy / scikit-learn** — Data processing and train/test split

---

## Getting Started

```bash
# 1. Install dependencies
pip install tensorflow mediapipe opencv-python numpy scikit-learn

# 2. Collect your own gesture data
python collect_data.py

# 3. Train the model
python train_model.py

# 4. Run the real-time translator
python app.py
```

---

## Model Architecture

| Layer | Type | Units |
|-------|------|-------|
| 1 | LSTM (return sequences) | 64 |
| 2 | LSTM (return sequences) | 128 |
| 3 | LSTM | 64 |
| 4 | Dense + Dropout (0.2) | 64 |
| 5 | Dense (Softmax) | 9 (classes) |

- Input shape: `(40 frames × 126 keypoints)`
- Optimizer: Adam · Loss: Categorical Crossentropy
- Early stopping: patience = 15 epochs

---

## Notes

- The model (`isl_custom_model.h5`) is trained on custom-collected data — accuracy improves with more diverse recordings.
- Confidence threshold for prediction display: **85%**
- Webcam index defaults to `1` — change to `0` if using a built-in camera.

---

## Contributing

Want to add more signs? Run `collect_data.py` with your name, record sequences, retrain with `train_model.py`, and open a pull request!
