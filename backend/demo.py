# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:43:11 2024

@author: raji
"""
import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
import time

import dlib
from imutils import face_utils
import matplotlib.pyplot as plt

# Function to return a dictionary of temperatures using OCR
def ocr_temp(image):
    # Get the image
    img = Image.open(image)
    width, height = img.size
    
    # Crop only the temp area (adjust crop_size as needed for your images)
    crop_size = (280, 0, width*0.40, height*0.25)
    crop_image = img.crop(crop_size)
    
    crop_image.save("temp_corner.jpeg")
    result_dict = {}
    try:
        # OCR the image
        ocr_img = np.array(crop_image)
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        text = pytesseract.image_to_string(ocr_img)
        print(text)

        # Convert text to a dictionary
        temp_list = text.replace("\n", " ").split()
        result_dict = {
            "Cen": float(temp_list[1]),
            "Max": float(temp_list[3]),
            "Min": float(temp_list[5])
        }
    except:
        result_dict = {
            "Cen": 35,
            "Max": 36.5,
            "Min": 27.5
        }
    
    return result_dict
    
    
# Initialize video capture (e.g., from an RTMP stream)
cap = cv2.VideoCapture("obs url")

# Define video codec and create VideoWriter object to save the stream
# Codec: 'XVID', 'MJPG', etc. 
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use 'MJPG' or 'MP4V' for other formats
out = cv2.VideoWriter('output_video.avi', fourcc, 1.0, (int(cap.get(3)), int(cap.get(4))))

# load the face detector (HOG-SVM)
#print("[INFO] loading dlib thermal face detector...")
detector = dlib.simple_object_detector("dlib_face_detector.svm")

# load the facial landmarks predictor
#print("[INFO] loading facial landmark predictor...")
predictor = dlib.shape_predictor("dlib_landmark_predictor.dat")


# List to store the temperatures for each detected face and risk and store the image
face_temperatures = {}
face_risks = {}
face_image = None

# for calculating the risk
threshold = 0.50
border = 0.80

# count the frames
frame_num = 0
wait_sec = 5
frame_rate = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # count the frames
    frame_num = frame_num + 1
    
    # skip the frame
    if frame_num % (wait_sec * frame_rate) != 0:
        continue
    
    # current frame
    print('----------')
    print('current frame: ',frame_num)
         
    # Assuming a thermal image is obtained from the video stream
    thermal_image = frame  # Use frame directly if it is a thermal image
    
    # Save the frame temporarily to apply OCR
    temp_image_path = 'temp_frame.jpeg'
    cv2.imwrite(temp_image_path, thermal_image)
    
    # Get the temperature using OCR function
    get_temp = ocr_temp(temp_image_path)

    # Assign the temperature min and max
    min_temp = get_temp['Min']  
    max_temp = get_temp['Max']  

    # Find the actual minimum and maximum grayscale values in the image
    min_pixel_value = np.min(thermal_image)
    max_pixel_value = np.max(thermal_image)

    # Normalize grayscale values to temperature
    temperature_map = min_temp + ((thermal_image - min_pixel_value) * (1 / (max_pixel_value - min_pixel_value))) * (max_temp - min_temp)
    
    # copy the image
    image_copy = thermal_image.copy()

    # convert the image to grayscale
    imagec = cv2.cvtColor(thermal_image, cv2.COLOR_BGRA2GRAY)
    
    # detect faces in the image
    rects = detector(imagec, upsample_num_times=1)
    
    if len(rects)==0:
        continue 

    face_key = 0
    for rect in rects:
        face_key = face_key+1
        # convert the dlib rectangle into an OpenCV bounding box and
        # draw a bounding box surrounding the face
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Crop the face region
        cropped_face = temperature_map[y:y+h, x:x+w]

        # Find the maximum temperature in the cropped face region
        max_face_temp = np.max(cropped_face)
        #face_temperatures.append{count,max_face_temp}
        
        if face_key not in face_temperatures:
            face_temperatures[face_key] = []
        
        if face_key not in face_risks:
            face_risks[face_key] = []
            
        face_temperatures[face_key].append(max_face_temp)
        
        # Prepare the temperature text
        temp_text = f"P: {face_key}, T: {max_face_temp:.1f} C"

        # Put the temperature text above the face
        cv2.putText(image_copy, temp_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # predict the location of facial landmark coordinates,
        # then convert the prediction to an easily parsable NumPy array
        shape = predictor(imagec, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates from our dlib shape
        # predictor model draw them on the image
        for (sx, sy) in shape:
            cv2.circle(image_copy, (sx, sy), 2, (0, 0, 255), -1)
            
        # Calculate risk  
        risk = ""
        if len(face_temperatures[face_key]) == 5:
            print(face_temperatures[face_key])
            tl = face_temperatures[face_key]
            delta = ((tl[1]-tl[0]) + (tl[2]-tl[1]) + (tl[3]-tl[2]) + (tl[4]-tl[3]))/4.0 
            if delta > threshold and delta >= border:
                print("Person: " + str(face_key) + ", High Risk")
                risk = 'High Risk'
            elif delta >= threshold:
                print("Person: " + str(face_key) + ", Moderate Risk")
                risk = 'Moderate Risk'
            else:
                print("Person: " + str(face_key) + ", No Risk")
                risk = 'No Risk'
            face_risks[face_key] = risk
            face_temperatures[face_key] = []

    # Save the current detected image for API
    face_image = image_copy.copy()
    
    # Write the frame to the video file
    out.write(image_copy)
    
    # Display the result
    cv2.imshow('OBS Stream with Face Temperature Detection', image_copy)
    
    # display the frame in console as plot
    plt.imshow(image_copy)
    plt.title("Detected Faces with Maximum Temperatures")
    plt.show()
    
    # Write face risks to a text file for api
    with open('face_risks.txt', 'w') as file:
        file.write(str(face_risks))
    
    # save the image for the api
    if face_image is not None:
        cv2.imwrite("api_image.jpeg", face_image)
        
    # print the face temp and risk
    print(face_temperatures)
    print(face_risks)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

