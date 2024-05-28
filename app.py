from flask import Flask, render_template,jsonify, Response, request
from number_plate_recognition import recognize_plate
import cv2
import csv
import pandas as pd
import os
import signal
from time import sleep


file_path_ = 'tfl (1)/tfl/uploads/cars.csv'


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames(file_path = file_path_):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640) # width of camera frame
    cap.set(4, 400) # height
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    with open(file_path, mode='a', newline='') as csvfile:
        # Create a DictWriter object, specifying the fieldnames
        fieldnames = ['Timestamp', 'CarNumber']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_with_plate, c_time, car_number = recognize_plate(frame)

        if frame_with_plate is not None:
            _, buffer = cv2.imencode('.jpg', frame_with_plate)
            frame_with_plate = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_with_plate + b'\r\n\r\n')

        if car_number is not None:
            # Write data to CSV only if car number is recognized
            data = {'Timestamp': c_time, 'CarNumber': car_number}
            writer.writerow(data)
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    if request.method == 'POST':
        shutdown_server()
        return 'Server shutting down...'
    else:
        return 'Invalid request method.'

def shutdown_server():
    pid = os.getpid()
    os.kill(pid, signal.SIGINT)

@app.route('/get_csv_data')
def get_csv_data():
    # Load the CSV file into a DataFrame
    df = pd.read_csv('tfl (1)/tfl/uploads/cars.csv')
    # Convert DataFrame to JSON
    data = df.to_dict(orient='records')
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)