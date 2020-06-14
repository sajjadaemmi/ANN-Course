import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('animals.h5')
video = cv2.VideoCapture('video.mp4')

while True:

    _, frame = video.read()
    cv2.imshow('sajjad', frame)

    # preprocess
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (299, 299))
    frame = frame / 255.0
    frame = frame.reshape(1, 299, 299, 3)

    y_pred = model.predict(frame)
    print(np.argmax(y_pred))

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()