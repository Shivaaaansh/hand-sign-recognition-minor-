import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
counter = 0
folder = r"C:\Desktop\sign gesture detection\Data\D"  # Using raw string for path

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white image for the final result
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure the crop region is valid
        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]
        
        # Check if the cropped image is empty
        if imgCrop.size == 0:
            print("Error: imgCrop is empty.")
            continue  # Skip this iteration or handle it accordingly

        imgCropShape = imgCrop.shape
        print(f"imgCrop Shape: {imgCropShape}")

        aspectratio = h / w
        print(f"Aspect Ratio: {aspectratio}")

        if aspectratio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wGap + wCal] = imgResize  # Ensure widths match

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hGap + hCal, :] = imgResize 

        # Display the cropped image and the white canvas with resized image
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    # Display the main video feed
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        filename = f'{folder}/Image_{time.time()}.jpg'
        cv2.imwrite(filename, imgWhite)
        print(f"Saved image as {filename}")
        print(f"Counter: {counter}")