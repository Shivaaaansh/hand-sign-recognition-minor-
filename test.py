import cv2
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import numpy as np
import math


def load_custom_model(model_path):
    from tensorflow.keras.layers import DepthwiseConv2D
    class CustomDepthwiseConv2D(DepthwiseConv2D):
        def __init__(self, **kwargs):
            kwargs.pop('groups', None)  
            super().__init__(**kwargs)

    custom_objects = {"DepthwiseConv2D": CustomDepthwiseConv2D}
    return load_model(model_path, custom_objects=custom_objects)


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)


try:
    model = load_custom_model("C:\Desktop\converted_keras (1)\keras_model.h5")
    with open("C:\Desktop\converted_keras (1)\labels.txt", "r") as f:
        labels = f.read().splitlines()
except Exception as e:
    print(f"Error loading model or labels: {e}")
    exit()


offset = 20
imgSize = 300  

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        
        imgCrop = img[max(0, y - offset): y + h + offset, max(0, x - offset): x + w + offset]

        if imgCrop.size == 0:
            print("Error: imgCrop is empty.")
            continue

        aspectRatio = h / w

       
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        
        imgWhite = cv2.resize(imgWhite, (224, 224))  

        
        imgWhite = imgWhite / 255.0  
        imgWhite = np.expand_dims(imgWhite, axis=0)  

        
        predictions = model.predict(imgWhite)
        index = np.argmax(predictions)
        print(predictions, index)

        
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                      (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (0, 255, 0), 4)

        
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite[0])  

    
    cv2.imshow('Image', imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()

