# Number Plate Detection Webpage

! [Number Plate recognition web page](uploads/ANPR_screen.png)

## Tech Stack
## YOLO11n.pt Training
- **[Colab Notebook](https://colab.research.google.com/drive/1NjlRSHjfrlHEKHN3rAELBoO7X087SxLG?usp=sharing)**
### Computer Vision
- **OpenCV**: Used for opening the camera, capturing frames, converting images to grayscale, drawing rectangles, and displaying text on images.
- **Haar Cascade Classifier**: Specifically, `haarcascade_russian_plate_number.xml` for detecting number plates.

### OCR (Optical Character Recognition)
- **Pytesseract**: Used for extracting text from the detected number plates.
  - **Parameter Tuning**:
    ```python
    custom_config = r'--oem 3 --psm 1'
    ```
  - **OCR Engine Mode (`--oem`)**:
    - `0`: Legacy engine only.
    - `1`: Neural nets LSTM engine only.
    - `2`: Legacy + LSTM engines.
    - `3`: Default, based on what is available.
  - **Page Segmentation Mode (`--psm`)**:
    - `0`: Orientation and script detection (OSD) only.
    - `1`: Automatic page segmentation with OSD.
    - `2`: Automatic page segmentation, but no OSD, or OCR.
    - `3`: Fully automatic page segmentation, but no OSD.
    - `4`: Assume a single column of text of variable sizes.
    - `5`: Assume a single uniform block of vertically aligned text.
    - `6`: Assume a single uniform block of text.
    - `7`: Treat the image as a single text line.
    - `8`: Treat the image as a single word.
    - `9`: Treat the image as a single word in a circle.
    - `10`: Treat the image as a single character.
    - `11`: Sparse text. Find as much text as possible in no particular order.
    - `12`: Sparse text with OSD.
    - `13`: Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

### Text Filtering
- **Regular Expressions (`re`)**: Used to filter out non-alphanumeric characters from the recognized text.

### Web Application
- **Flask**: Used for web application framework to render templates, jsonify responses, and handle requests.
  - Key functions: `render_template`, `jsonify`, `Response`, `request`.

### Data Storage
- **CSV Library**: Used to append detected number plates to a CSV file for record-keeping.

### Data Display
- **Pandas**: Used for loading and displaying data on the webpage after being converted to JSON format.

### Interrupt Handling
- **Signal**: Used to send a SIGINT signal to interrupt the current Python process.

### Timing Control
- **Time**: Used for controlling the frame capture timing until a number plate is recognized.

## Implementation Logic

1. **Camera Initialization and Frame Capture**:
   - The camera opens and starts capturing frames.
   - Each captured frame is converted to grayscale for processing.

    ```python
    import cv2

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Further processing
    ```

2. **Number Plate Detection**:
   - The grayscale frame is sent to the recognition engine using the Haar Cascade Classifier to detect number plates.
   - If a number plate is detected, a rectangle is drawn around it, and a label is put with the recognized text.
   - If no number plate is detected, no label is put.

    ```python
    plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 1)
        # Further processing in source file

3. **Displaying on screen**:
    ```python
    Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

4. **Saving to CSV**


5. **Interrupt Handling**

6. **Timing Control**:
   - The `time` module is used to pause the frame capture until a number plate is detected.
=======
# Car-Plate-detection
Computer Vision Implementation with PyTesseract and Haar Cascade  This repository features a computer vision project using PyTesseract and Haar Cascade to detect number regions and recognize characters. Built with Flask, it provides a web interface for user interaction. Detected numbers are stored in a CSV file with timestamps.
