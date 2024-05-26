from flask import Flask, render_template, Response
from number_plate_recognition import recognize_plate
import cv2
import pandas as pd

cars = pd.read_csv('tfl (1)/tfl/uploads/cars.csv')


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640) # width of camera frame
    cap.set(4, 480) # height
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_with_plate = recognize_plate(frame)  # Assuming recognize_plate now takes a frame as input

        _, buffer = cv2.imencode('.jpg', frame_with_plate)
        frame_with_plate = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_with_plate + b'\r\n\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)