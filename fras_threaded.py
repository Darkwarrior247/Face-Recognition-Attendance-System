import os
import json
import numpy as np
import face_recognition
import cv2
import datetime
import logging
import sys
import time
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('attendance_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class UIOverlay:
    def __init__(self, config):
        self.config = config
        self.message = ""
        self.show_checkmark = False
        self.display_until = 0

    def set_message(self, message, show_checkmark=False, duration=2):
        self.message = message
        self.show_checkmark = show_checkmark
        self.display_until = time.time() + duration

    def should_clear(self):
        return self.display_until and time.time() > self.display_until

    def clear(self):
        self.message = ""
        self.show_checkmark = False
        self.display_until = 0

def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        config = {
            "frame_skip": 2,
            "face_recognition_threshold": 0.50,
            "attendance_file": "attendance.csv",
            "ui": {
                "analyzing_text": "Analyzing...",
                "welcome_text": "Welcome,",
                "unknown_text": "Unknown Person",
                "display_time": 2
            }
        }
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)
        return config

class FaceRecognitionSystem:
    def __init__(self, config):
        self.config = config
        self.classNames = []
        self.encodeListKnown = []
        self.ui_overlay = UIOverlay(config)
        self.state = "idle"  # idle, analyzing, welcome, unknown, cooldown
        self.state_until = 0
        self.last_detected_name = None
        self.load_encodings()
        # Threading additions
        self.recognition_thread = None
        self.recognition_result = None
        self._pending_face_locations = None
        self._pending_rgb_img = None

    def load_encodings(self):
        encoding_file = 'encodings.json'
        if not os.path.exists(encoding_file):
            logging.error(f"{encoding_file} not found.")
            return False
        with open(encoding_file, 'r') as f:
            data = json.load(f)
        self.classNames = list(data.keys())
        self.encodeListKnown = [np.array(enc) for enc in data.values()]
        logging.info(f"Loaded {len(self.classNames)} face encodings.")
        return True

    def markAttendance(self, name):
        try:
            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            if not os.path.exists(self.config['attendance_file']):
                with open(self.config['attendance_file'], 'w') as f:
                    f.write("Name,Date,Time\n")
            with open(self.config['attendance_file'], 'r+') as f:
                myDataList = f.readlines()
                today_attendance = [
                    line for line in myDataList 
                    if name in line and current_date in line
                ]
                if not today_attendance:
                    f.write(f"{name},{current_date},{current_time}\n")
                    logging.info(f"Attendance marked for {name} at {current_date} {current_time}")
                    return True
                else:
                    logging.info(f"Attendance already marked for {name} today.")
            return False
        except Exception as e:
            logging.error(f"Error marking attendance for {name}: {str(e)}", exc_info=True)
            return False

    def draw_ui(self, img):
        if not self.ui_overlay.message:
            return img
        height, width = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, height-100), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.putText(img, self.ui_overlay.message,
                    (20, height-40), cv2.FONT_HERSHEY_DUPLEX,
                    1.0, (255, 255, 255), 2)
        if self.ui_overlay.show_checkmark:
            cv2.putText(img, "✔", (width-60, height-40), cv2.FONT_HERSHEY_DUPLEX,
                        2.0, (0, 255, 0), 3)
        return img

    def start_recognition_thread(self, rgb_img, face_locations):
        def recognize():
            name = "Unknown"
            found_encoding = False
            if face_locations and rgb_img is not None:
                face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                if face_encodings:
                    found_encoding = True
                    matches = face_recognition.compare_faces(
                        self.encodeListKnown, face_encodings[0], tolerance=self.config['face_recognition_threshold']
                    )
                    face_distances = face_recognition.face_distance(self.encodeListKnown, face_encodings[0])
                    best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

                    if best_match_index is not None and matches[best_match_index]:
                        name = self.classNames[best_match_index]
            self.recognition_result = (name, found_encoding)
        self.recognition_result = None
        self.recognition_thread = threading.Thread(target=recognize)
        self.recognition_thread.start()

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Failed to open webcam.")
            return

        logging.info("Face Recognition Attendance System started.")
        try:
            while True:
                try:
                    success, img = cap.read()
                    if not success:
                        logging.error("Failed to read frame from webcam.")
                        break

                    now = time.time()

                    # State machine
                    if self.state == "idle":
                        small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
                        rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_small_img)
                        if face_locations:
                            self.state = "analyzing"
                            self.state_until = now + 1.0  # 1 second analyzing
                            self.ui_overlay.set_message(self.config['ui']['analyzing_text'], False, 1.0)
                            self._pending_face_locations = face_locations
                            self._pending_rgb_img = rgb_small_img
                            self.start_recognition_thread(rgb_small_img, face_locations)
                        # else: remain idle, no overlay

                    elif self.state == "analyzing":
                        if now >= self.state_until:
                            if self.recognition_thread and not self.recognition_thread.is_alive():
                                name, found_encoding = self.recognition_result if self.recognition_result else ("Unknown", False)
                                if found_encoding:
                                    if name != "Unknown":
                                        self.markAttendance(name)
                                        self.ui_overlay.set_message(
                                            f"{self.config['ui']['welcome_text']} {name}",
                                            True,
                                            self.config['ui']['display_time']
                                        )
                                        self.state = "welcome"
                                        self.state_until = now + self.config['ui']['display_time']
                                        self.last_detected_name = name
                                    else:
                                        self.markAttendance("Unknown")
                                        self.ui_overlay.set_message(
                                            self.config['ui']['unknown_text'],
                                            False,
                                            2
                                        )
                                        self.state = "unknown"
                                        self.state_until = now + 2
                                else:
                                    # No encoding found, treat as unknown
                                    self.ui_overlay.set_message(
                                        self.config['ui']['unknown_text'],
                                        False,
                                        2
                                    )
                                    self.state = "unknown"
                                    self.state_until = now + 2
                            else:
                                # Recognition still running, wait a bit more
                                self.state_until = now + 0.1

                    elif self.state in ("welcome", "unknown"):
                        if now >= self.state_until:
                            self.state = "cooldown"
                            self.state_until = now + 2  # 2 seconds cooldown after any detection
                            self.ui_overlay.clear()

                    elif self.state == "cooldown":
                        if now >= self.state_until:
                            self.state = "idle"

                    # Draw rectangles if face detected
                    if hasattr(self, "_pending_face_locations") and self.state in ("analyzing", "welcome", "unknown"):
                        for (top, right, bottom, left) in self._pending_face_locations:
                            top, right, bottom, left = top*4, right*4, bottom*4, left*4
                            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                            if self.state == "welcome" and self.last_detected_name:
                                cv2.putText(img, self.last_detected_name, (left, top-10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                            elif self.state == "unknown":
                                cv2.putText(img, "Unknown", (left, top-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

                    # UI overlay
                    if self.ui_overlay.should_clear():
                        self.ui_overlay.clear()
                    img = self.draw_ui(img)

                    cv2.imshow('Face Recognition Attendance', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logging.info("User requested exit with 'q'.")
                        break

                except Exception as frame_err:
                    logging.error(f"Error during frame processing: {frame_err}", exc_info=True)
                    continue  # Optionally skip to next frame

        except KeyboardInterrupt:
            logging.info("Application terminated by user (Ctrl+C).")
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}", exc_info=True)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logging.info("Camera released and all windows closed.")

if __name__ == "__main__":
    try:
        logging.info("Application starting.")
        config = load_config()
        frs = FaceRecognitionSystem(config)
        frs.run()
        logging.info("Application exited normally.")
    except Exception as e:
        logging.error(f"Fatal error on startup: {e}", exc_info=True)