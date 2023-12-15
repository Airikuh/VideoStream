from keras.models import load_model
from time import sleep
#edited here
#from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + \
'haarcascade_frontalface_default.xml')
path_Model = r'C:\Users\airik\Desktop\MobileNet_model2.h5'
classifier =load_model(path_Model)

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# input file
#edited here
filename = r'C:\Users\airik\Desktop\CNN_1_852x480.mp4'

cap = cv2.VideoCapture(filename)
if (cap.isOpened() == False): 
        print("Error reading video file")

    # output file
    # We need to set resolutions. So, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
  
    # Below VideoWriter object will create a frame of above defined The output 
    # is stored in 'output.mp4' file.
result = cv2.VideoWriter('output.mp4', 
                             0x00000021,
                             10, size)
    
while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        img_RGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            if w < frame_height/10:
                continue   # skip small features

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_RGB = img_RGB[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            roi_RGB = cv2.resize(roi_RGB,(224,224),interpolation=cv2.INTER_AREA)           

            if np.sum([roi_gray])!=0:
                #roi = roi_gray.astype('float')/255.0
                roi = roi_RGB.astype('float')/255.0
                #roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Emotion Detector',frame)
        result.write(frame)  # for output file 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
