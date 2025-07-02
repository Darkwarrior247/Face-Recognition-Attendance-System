# Face Recognition Attendance System

## 📋 Overview
The Face Recognition Attendance System is a real-time, privacy-focused solution for automated attendance tracking using facial recognition. It leverages computer vision and machine learning to detect and recognize faces from a webcam, logging attendance with timestamps in a secure and efficient manner.

- **No images are stored**—only face encodings are kept for privacy and speed.
- **Unknown faces** are also logged (once per day) for audit purposes.
- Robust error handling and logging ensure reliability in real-world use.

---

## ✨ Key Features

- Real-time face detection and recognition
- Automated attendance logging with timestamps
- User-friendly interface with visual feedback overlays
- Multi-face detection capability
- Privacy-focused: only encodings, not images, are stored
- Configurable system parameters via `config.json`
- Comprehensive error logging to `attendance_system.log`
- Logs both known and unknown faces (once per day)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.6+
- Visual Studio Build Tools with C++ (for dlib/face_recognition)
- CMake
- Webcam (720p minimum)
- 4GB+ RAM

### Installation

1. **Install Visual Studio Build Tools**
   - Download Visual Studio Community Edition
   - Select "Desktop Development with C++"
   - Complete installation and restart

2. **Clone the repository**
   ```bash
   git clone https://github.com/lifaet/Face-Recognition-Attendance-System.git
   cd Face-Recognition-Attendance-System
   ```

3. **Install Python dependencies**
   ```bash
   pip install cmake
   pip install dlib
   pip install face-recognition
   pip install numpy
   pip install opencv-python
   ```

---

## 📂 Project Structure

```
Face-Recognition-Attendance-System/
├── fras.py                 # Main application
├── add_attendees.py        # Register new attendees (encodings only)
├── encodings.json          # Stores face encodings and names
├── attendance.csv          # Attendance records
├── config.json             # System configuration
├── attendance_system.log   # System logs
└── README.md               # Documentation
```

---

## ⚙️ Configuration

The system is configured via `config.json`:

```json
{
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
```

- `frame_skip`: Process every Nth frame for speed.
- `face_recognition_threshold`: Lower is stricter (default 0.50).
- `display_time`: Seconds to show welcome message.

---

## 📸 Registering Attendees

Use `add_attendees.py` to add new people to the system. This script captures a face from the webcam, extracts its encoding, and saves it to `encodings.json`.

### Usage

1. Run the script:
   ```bash
   python add_attendees.py
   ```
2. Enter the attendee's name.
3. Position their face in the webcam and press `c` to capture.
4. The encoding is saved to `encodings.json`.

**Note:** No images are stored—only the encoding is kept for privacy.

---

## 🖥️ Running the Attendance System

1. Start the application:
   ```bash
   python fras.py
   ```
2. The webcam opens and waits for a face.
3. When a face is detected:
   - Shows "Analyzing..." for 1–2 seconds.
   - If recognized, shows "Welcome, NAME!" and logs attendance.
   - If not recognized, shows "Unknown Person" and logs "Unknown" attendance.
4. Press `q` or `Ctrl+C` to exit.

---

## 🔍 Technical Details

### Face Recognition Process

1. **Face Detection**
   - Locates faces in video feed
   - Processes at 1/4 resolution for performance

2. **Face Recognition**
   - Converts detected faces to encodings
   - Matches against known face encodings from `encodings.json`
   - Threshold-based verification

3. **Attendance Marking**
   - Automatic date and time stamping
   - Duplicate entry prevention (one entry per person per day)
   - CSV format storage

### UI Features

- Status messages for:
  - Face analysis in progress
  - Welcome messages
  - Unknown person alerts
- Visual indicators:
  - Green rectangle around detected faces
  - Checkmark for successful recognition
- Semi-transparent overlay

---

## 📊 Performance

- Frame skipping for optimal performance (`frame_skip` in config)
- Configurable recognition threshold
- Efficient image processing and memory usage

---

## 🔧 Error Handling & Logging

- All errors and events are logged to `attendance_system.log`.
- Graceful shutdown on errors or `Ctrl+C`.
- Frame-level errors are logged and skipped without crashing the system.

---

## 📝 Attendance Logging

- Attendance is logged in `attendance.csv` as:
  ```
  Name,Date,Time
  ```
- Each person (including "Unknown") is logged only once per day.

---

## 🎯 Applications

### Educational
- Classroom attendance
- Event tracking
- Library access

### Corporate
- Employee attendance
- Meeting participation
- Visitor tracking

### Events
- Participant check-in
- Access control
- Session tracking

---

## 🔧 Troubleshooting

### Common Issues

1. **Camera not detected**
   - Check camera connections
   - Verify camera permissions

2. **Recognition issues**
   - Ensure good lighting
   - Update reference encodings
   - Adjust recognition threshold

3. **Performance issues**
   - Increase frame skip value
   - Check system resources
   - Update hardware drivers

---

## 🔒 Security & Privacy

- Only face encodings (not images) are stored.
- No images are saved after registration.
- Attendance logs do not include images or biometric data, only names and timestamps.

---

## 📜 License

This project is for educational and personal use.  
