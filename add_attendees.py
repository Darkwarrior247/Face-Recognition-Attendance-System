import cv2
import face_recognition
import os
import json
from pathlib import Path
import numpy as np

def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Config file not found. Creating default configuration...")
        default_config = {
            "path": "attendees",
            "frame_skip": 2,
            "face_recognition_threshold": 0.50,
            "attendance_file": "attendance.csv",
            "ui": {
                "analyzing_text": "Analyzing...",
                "welcome_text": "Welcome,",
                "unknown_text": "Unknown Person",
                "display_time": 3
            }
        }
        try:
            with open('config.json', 'w') as f:
                json.dump(default_config, f, indent=4)
            print("Default configuration created successfully.")
            return default_config
        except Exception as e:
            print(f"Error creating config file: {str(e)}")
            return None

def capture_training_data():
    config = load_config()
    if not config:
        return

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Failed to open webcam")
            return

        person_name = input("Enter person's name (or 'q' to quit): ").strip()
        if person_name.lower() == 'q':
            return

        person_name = person_name.replace(" ", "_").upper()

        print("\nInstructions:")
        print("1. Position face in the green box")
        print("2. Press 'c' to capture")
        print("3. Press 'q' to quit\n")

        while True:
            success, raw_frame = cap.read()
            if not success:
                print("Error: Failed to read frame")
                break

            display_frame = raw_frame.copy()
            height, width = display_frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            box_size = 300

            cv2.rectangle(display_frame, 
                         (center_x - box_size//2, center_y - box_size//2),
                         (center_x + box_size//2, center_y + box_size//2),
                         (0, 255, 0), 2)

            cv2.putText(display_frame, "Position face in box and press 'c' to capture",
                       (20, height-40), cv2.FONT_HERSHEY_DUPLEX,
                       0.7, (255, 255, 255), 2)

            cv2.imshow('Capture Training Image', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                face_locations = face_recognition.face_locations(raw_frame)
                if len(face_locations) == 0:
                    print("No face detected. Please try again.")
                    continue
                elif len(face_locations) > 1:
                    print("Multiple faces detected. Please ensure only one face is visible.")
                    continue

                # Extract encoding and save to encodings.json
                rgb_img = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_img, face_locations)
                if not encodings:
                    print("No face encoding found. Please try again.")
                    continue
                encoding = encodings[0].tolist()

                encoding_file = 'encodings.json'
                if os.path.exists(encoding_file):
                    with open(encoding_file, 'r') as f:
                        data = json.load(f)
                else:
                    data = {}

                data[person_name] = encoding

                with open(encoding_file, 'w') as f:
                    json.dump(data, f)

                print(f"\nEncoding for {person_name} saved successfully.")
                break

    except Exception as e:
        print(f"Error capturing training data: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def list_attendees():
    encoding_file = 'encodings.json'
    if os.path.exists(encoding_file):
        with open(encoding_file, 'r') as f:
            data = json.load(f)
        attendees = list(data.keys())
        if attendees:
            print("\nRegistered Attendees:")
            for i, name in enumerate(attendees, 1):
                print(f"{i}. {name}")
        else:
            print("\nNo attendees registered yet.")
    else:
        print("\nNo encodings file found.")

def delete_attendee():
    encoding_file = 'encodings.json'
    if not os.path.exists(encoding_file):
        print("\nNo encodings file found.")
        return

    with open(encoding_file, 'r') as f:
        data = json.load(f)
    attendees = list(data.keys())
    if not attendees:
        print("\nNo attendees to delete.")
        return

    print("\nSelect attendee to delete:")
    for i, name in enumerate(attendees, 1):
        print(f"{i}. {name}")

    try:
        choice = int(input("\nEnter number (or 0 to cancel): "))
        if choice == 0:
            return
        if 1 <= choice <= len(attendees):
            name_to_delete = attendees[choice-1]
            del data[name_to_delete]
            with open(encoding_file, 'w') as f:
                json.dump(data, f)
            print(f"\nDeleted: {name_to_delete}")
        else:
            print("\nInvalid selection.")
    except ValueError:
        print("\nInvalid input. Please enter a number.")

def add_attendee(name, image_path, encoding_file='encodings.json'):
    # Load image and get encoding
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img)
    if not encodings:
        print("No face found in the image.")
        return
    encoding = encodings[0].tolist()  # Convert numpy array to list for JSON

    # Load existing encodings
    if os.path.exists(encoding_file):
        with open(encoding_file, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # Add/update encoding
    data[name] = encoding

    # Save back to file
    with open(encoding_file, 'w') as f:
        json.dump(data, f)

    print(f"Encoding for {name} saved.")

def main():
    try:
        while True:
            print("\nFace Recognition Training System")
            print("================================")
            print("1. Add New Person")
            print("2. View Registered Attendees")
            print("3. Delete Attendee")
            print("4. Exit")
            
            choice = input("\nSelect option: ")
            
            if choice == "1":
                capture_training_data()
            elif choice == "2":
                list_attendees()
            elif choice == "3":
                delete_attendee()
            elif choice == "4":
                print("\nExiting training system...")
                break
            else:
                print("\nInvalid option. Please try again.")

    except KeyboardInterrupt:
        print("\nTraining system terminated by user")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()