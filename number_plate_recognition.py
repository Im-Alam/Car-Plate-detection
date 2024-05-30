import cv2
import pytesseract
from datetime import datetime
import re


def recognize_plate(frame):
    harcascade = 'haarcascade_russian_plate_number.xml'
    plate_cascade = cv2.CascadeClassifier(harcascade)

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    c_time = None
    ret_txt = None

    # find the coordinate for plates in the image
    plates = plate_cascade.detectMultiScale(img_gray, 1.05, 8)  # Adjust parameters as needed

    for (x, y, w, h) in plates:
        area = w * h
        min_area = 1000
        if area > min_area:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 1)
            #cv2.putText(frame, 'Number Plate', (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
            img_roi = frame[y:y+h, x:x+w]
            
            try:
                # Use of OCR
                custom_config = r'--oem 3 --psm 1'  # Adjust OCR parameters as needed
                outputT = pytesseract.image_to_string(img_roi, config=custom_config)
                alphanumeric_text = re.sub(r'[^a-zA-Z0-9]', '', outputT)
                min_len = 1
                max_len = 10
                if min_len<=len(alphanumeric_text)<=max_len:
                    cv2.putText(frame, outputT, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
                    c_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    return frame, c_time, alphanumeric_text
                
                
            except Exception as e:
                print(f"Error in OCR: {e} ++++++++++++++++++++++++++++++")
    return frame, c_time, ret_txt

 
