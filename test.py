import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'D:\Python Projects\Facial Expressions Recognition\haarcascade_frontalface_default.xml')
model = load_model(r'D:\Python Projects\Facial Expressions Recognition\Emotion_little_vgg.h5')

class_labels = ['Angry','Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    labels=[]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float32') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predicting the emotion
            preds = model.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow('Facial Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# This code captures video from the webcam, detects faces, and predicts the emotion of the detected faces using a pre-trained model.
# It uses OpenCV for face detection and Keras for emotion classification.
# The model is loaded from a specified path, and the Haar Cascade classifier is used for face detection.
# The code also handles the case where no face is found by displaying a message on the screen.
# The predictions are displayed on the video feed in real-time.
# The program exits when the 'q' key is pressed.
# Ensure that the paths to the Haar Cascade XML file and the model are correct.
# The model should be trained on the same dataset as the one used for training the model.


