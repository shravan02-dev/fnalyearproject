from flask import Flask, request, render_template, send_file
import os
import cv2
import tempfile
import face_recognition
from tracker import Tracker
from ultralytics import YOLO
import random
from moviepy.editor import *
import sqlite3

app = Flask(__name__)
def create_tables():
    conn = sqlite3.connect('tracks.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS tracks (
                        id INTEGER PRIMARY KEY,
                        track_id INTEGER,
                        name TEXT,
                        frame_time_seconds REAL
                    )''')
    conn.commit()
    conn.close()

# Call the function to create tables when the Flask app starts
create_tables()

def insert_track_info(track_id, frame_time_seconds, name):
    conn = sqlite3.connect('tracks.db')
    cursor = conn.cursor()

    # Check if the record already exists
    cursor.execute('''SELECT * FROM tracks WHERE track_id=? AND name=? AND frame_time_seconds=?''', (track_id, name, frame_time_seconds))
    existing_record = cursor.fetchone()

    # If the record does not exist, insert it into the table
    if not existing_record:
        cursor.execute('''INSERT INTO tracks (track_id, name, frame_time_seconds) VALUES (?, ?, ?)''', (track_id, name, frame_time_seconds))
        conn.commit()

    conn.close()
    if not existing_record:
     with open('track_data.txt', 'a') as file:
        file.write(f"Track ID: {track_id}, Name: {name}, Frame Time (seconds): {frame_time_seconds}\n")

# Function to process the uploaded video
def process_video(input_video_path, recognized_name, recognized_photo):
    
    # Your video processing code goes here
    known_face_encodings = []
    known_face_names = []
    video_out_path = os.path.join('.', 'out.mp4')
    # Load the known face encoding and name
        # Load the known face encoding and name
    photo_image = face_recognition.load_image_file(recognized_photo)
    face_encoding = face_recognition.face_encodings(photo_image)
    
    known_face_encodings.append(face_encoding[0])
    known_face_names.append(recognized_name)

    cap = cv2.VideoCapture(input_video_path)
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    # Initialize video capture
    
    frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 
    ret, frame = cap.read()

    # Initialize video writer
    cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

    # Initialize YOLO model
    model = YOLO("yolov8n.pt")

    # Initialize object tracker
    tracker = Tracker()

    # Generate random colors for bounding boxes
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

    # Detection threshold
    detection_threshold = 0.5
    id = 0
    
    # Main loop to process each frame
    while ret:
        # Perform object detection using YOLO
        results = model(frame)
        frame_number += 1
        
        # Calculate frame time in seconds
        frame_time = frame_number / frame_rate
        # Process each detection result
        for result in results:
            detections = []
            name = None
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if score > detection_threshold and class_id == 0:
                    detections.append([x1, y1, x2, y2, score])

            # Update tracker with detected objects
            tracker.update(frame, detections)
            
            # Draw bounding boxes and track IDs for each tracked object
            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[track_id % len(colors)], 2)
                
                # Recognize faces within the detected objects
                # Convert the detected face region to RGB color format
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                if id == track_id:
                    # Blur the background
                    blur_frame = frame.copy()
                    blur_frame[:y1, :] = cv2.GaussianBlur(blur_frame[:y1, :], (31, 31), 0)
                    blur_frame[y2:, :] = cv2.GaussianBlur(blur_frame[y2:, :], (31, 31), 0)
                    blur_frame[:, :x1] = cv2.GaussianBlur(blur_frame[:, :x1], (31, 31), 0)
                    blur_frame[:, x2:] = cv2.GaussianBlur(blur_frame[:, x2:], (31, 31), 0)
                    insert_track_info(track_id, frame_time, recognized_name)
                    alpha = 0.5
                    frame = cv2.addWeighted(blur_frame, alpha, frame, 1 - alpha, 0)
                    cv2.putText(frame, recognized_name, (int(x1), int(y1) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                   
                if 0 <= y1 < frame.shape[0] and 0 <= y2 < frame.shape[0] and 0 <= x1 < frame.shape[1] and 0 <= x2 < frame.shape[1]:
                    # Convert the detected face region to RGB color format
                    face_image = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
             
                    face_locations = face_recognition.face_locations(face_image)
                    
                    # Get the face encodings of the detected face
                    face_encodings = face_recognition.face_encodings(face_image, face_locations)
                    
                    # Check if face encodings are found
                    if face_encodings:
                        # Compare the detected face with known face encodings
                        matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])

                        # Check if any matches are found
                        if any(matches):
                            # Get the index of the first match
                            # Get the corresponding identified identity
                            # Draw the identity label on the frame
                            # Inside process_video function
                            
                            id = track_id 
                        else:
                            # No match found
                            cv2.putText(frame, "Unknown", (int(x1), int(y1) - 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
       
        # Write the processed frame to output video
        cap_out.write(frame)
        
        # Read the next frame
        ret, frame = cap.read()

    # Release video capture and writer
    cap.release()
    cap_out.release()
    cv2.destroyAllWindows()
    templates="C:\\Users\\HP\\Desktop\\object-tracking-yolov8-deep-sort - Copy - Copy - Copy - Copy (2) - Copy\\templates"
    input_file="out.mp4"
    output_file = "output_video"

    convert_to_mp4(input_file, output_file)
    return video_out_path

# Route to upload video file 
def convert_to_mp4(input_file, output_file):
    try:
        # Load the video file
        video_clip = VideoFileClip(input_file)

        # Set output file format to MP4
        output_file = output_file if output_file.endswith('.mp4') else output_file + '.mp4'

        # Write the video clip to a new file with MP4 format
        video_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')

        # Close the video clip
        video_clip.close()

        print(f"Conversion successful. MP4 file saved as {output_file}")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

@app.route("/", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template("index.html", message="No file part")
        
        file = request.files['file']
        # Get name from form
        name = request.form.get('name')
        photo = request.files['photo']  # Get photo from form
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template("index.html", message="No selected file")
        
        if file and photo:
            # Save the uploaded file to a temporary directory
            _, temp_file_path = tempfile.mkstemp(suffix=".mp4")
            file.save(temp_file_path)
            
            # Save the uploaded photo to a temporary directory
            _, temp_photo_path = tempfile.mkstemp(suffix=".png")
            photo.save(temp_photo_path)
            
            # Process the video
            video_path = process_video(temp_file_path, name, temp_photo_path) 


            # Return link to download the processed video
            return render_template("index.html", video_path=video_path)

    return render_template("index.html")

# Serve the processed video file
if __name__ == "__main__":
    app.run(debug=True)
