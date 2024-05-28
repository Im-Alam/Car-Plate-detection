import cv2
from openalpr import Alpr
from datetime import datetime


def recognize_plate(frame):
    harcascade = 'tfl (1)/tfl/haarcascade_russian_plate_number.xml'
    plate_cascade = cv2.CascadeClassifier(harcascade)
    alpr = Alpr("us", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data")

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    c_time = None
    outputT = None
    # find the coordinate for plates in the image
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)  # Adjust parameters as needed

    for (x, y, w, h) in plates:
        area = w * h
        min_area = 700
        if area > min_area:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 1)
            cv2.putText(frame, 'Number Plate', (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
            
            img_roi = frame[y:y+h, x:x+w]  # Corrected the variable name
            
            try:
                # Use easy for OCR
                custom_config = r'--oem 3 --psm 1'  # Adjust OCR parameters as needed
                outputT = None
                outputT = alpr.recognize(img_roi)
                cv2.putText(frame, outputT, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
                
                c_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                

            except Exception as e:
                print(f"Error in OCR: {e} ++++++++++++++++++++++++++++++")
    return frame, c_time, outputT

 