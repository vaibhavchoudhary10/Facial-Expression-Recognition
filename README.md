Dataset Link : https://www.kaggle.com/datasets/msambare/fer2013

# Facial Expressions Recognition

This project is a real-time facial emotion recognition system that uses a deep learning model trained within the project to classify emotions from facial expressions. The system captures video from a webcam, detects faces, and predicts the emotion of the detected faces.

## Features

- Real-time facial emotion recognition.
- Detects and classifies emotions into five categories: Angry, Happy, Sad, Surprise, and Neutral.
- Uses OpenCV for face detection and TensorFlow/Keras for emotion classification.
- Displays the predicted emotion on the video feed.

## Project Structure
```Emotion_little_vgg.h5``` # Trained model for emotion recognition ```haarcascade_frontalface_default.xml``` # Haar Cascade XML file for face detection ```test.py``` # Script for real-time emotion recognition ```training.py``` # Script for training the emotion recognition model test/ # Test dataset with categorized images train/ # Training dataset with categorized images


## Requirements

- Python 3.7 or higher
- TensorFlow
- Keras
- OpenCV
- NumPy

Install the required libraries using pip:

```sh
pip install tensorflow opencv-python numpy

```

## Usage
Training the Model
To train the model, run the training.py script. Ensure that the training and validation datasets are correctly placed in the train/ and test/ directories, respectively.
```sh
python [training.py](http://_vscodecontentref_/4)
```

The trained model will be saved as Emotion_little_vgg.h5.

Running the Emotion Recognition System
To run the real-time emotion recognition system, execute the test.py script. Ensure that the paths to the Haar Cascade XML file and the pre-trained model are correct.

```sh
python [test.py](http://_vscodecontentref_/5)
```

Press the q key to exit the program.

## Dataset
The dataset is organized into the following categories:

    Angry
    Happy
    Neutral
    Sad
    Surprise
Each category contains images for training and testing. Ensure that the dataset is preprocessed and resized to 48x48 grayscale images before training.

## Model Architecture
The model is a Convolutional Neural Network (CNN) with the following layers:

    1.Five convolutional layers with ELU activation, batch normalization, and max pooling.
    2.Two fully connected layers with dropout for regularization.
    3.A softmax output layer for emotion classification.

## Notes
Ensure that the paths to ```Emotion_little_vgg.h5``` and ```haarcascade_frontalface_default.xml``` are correct in the scripts.
The model should be trained on the same dataset used for testing to ensure compatibility.
The webcam must be connected and accessible for real-time emotion recognition.
