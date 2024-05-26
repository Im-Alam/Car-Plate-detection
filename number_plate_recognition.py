import cv2
import pytesseract
import time

pytesseract.pytesseract.tesseract_cmd = 'tesseract'


def recognize_plate(frame):
    harcascade = 'haarcascade_russian_plate_number.xml'
    plate_cascade = cv2.CascadeClassifier(harcascade)

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find the coordinate for plates in the image
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)  # Adjust parameters as needed

    for (x, y, w, h) in plates:
        area = w * h
        min_area = 100
        if area > min_area:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 1)
            cv2.putText(frame, 'Number Plate', (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
            
            img_roi = frame[y:y+h, x:x+w]  # Corrected the variable name
            
            try:
                # Use Tesseract for OCR
                custom_config = r'--oem 3 --psm 1'  # Adjust OCR parameters as needed

                output = pytesseract.image_to_string(img_roi, config=custom_config)
                cv2.putText(frame, output, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
                
                t = time.localtime()
                current_time = time.strftime("%H:%M:%S", t)
                
            except Exception as e:
                print(f"Error in OCR: {e}")
    return frame