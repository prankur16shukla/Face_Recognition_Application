# importing the libraries required
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

# loading the face recognition model
model = load_model('facerecognition_model.h5')
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# extracting face data from frame
def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    for (x, y, w, h) in faces:
        x = x-10
        y = y-10
        cropped_face = img[y:y+h+50, x:x+w+50]
        
    return cropped_face

# initializing the video frame
video_capture = cv2.VideoCapture(0)

# function to get face data and checking it for matches
while True:
    _, frame = video_capture.read()

    face = face_extractor(frame)

    if type(face) is np.ndarray:
    
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')

        img_array = np.array(im)

        img_array = np.expand_dims(img_array, axis = 0)
        pred = model.predict(img_array)
        print(pred)

        # predicting the result based on probability
        if (pred[0][0] > 0.5):
            name = "Ankur"
            cv2.putText(frame, name, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        elif (pred[0][1] > 0.5):
            name = "Prankur"
            cv2.putText(frame, name, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        elif (pred[0][2] > 0.5):
            name = "Rahul"
            cv2.putText(frame, name, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "No Match Found", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
        
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()