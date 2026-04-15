# Smart Face Attendance System

Streamlit-based attendance system using ArcFace (ONNX) + Mediapipe face detection.

## Features

- Face-based attendance marking with webcam
- Class and subject-based attendance sessions
- Student registration with photo capture
- Dashboard for daily attendance records
- Absentee list export (CSV)
- Optional absentee email notifications
- Basic management views for classes, subjects, and students

## Project Structure

- `app.py`: Main app entry point (current working app)
- `images/`: Stored student face images
- `models/`: ONNX models
- `students.csv`: Student master data
- `classes.csv`: Class list
- `subjects.csv`: Subject list
- `attendance.xlsx`: Generated attendance log

## Requirements

- Python 3.10+
- Webcam
- ONNX model file in `models/w600k_r50.onnx`

Install dependencies:

```bash
pip install streamlit opencv-python numpy onnxruntime pandas mediapipe pillow openpyxl
```

If you want GPU inference with ONNX Runtime, install GPU runtime instead of CPU runtime:

```bash
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
```

## Run

From the project root:

```bash
streamlit run app.py
```

## App Flow

1. Add classes and subjects in the management section.
2. Register students with class, email, and face image.
3. Start an attendance session by selecting class + subject.
4. Review absentees, manually mark present if needed, and optionally send emails.

## Configuration Notes

Email sender configuration is currently defined in code in `app.py` (`SENDER_EMAIL`, `SENDER_PASSWORD`).

Recommended next step:

- Move email credentials to environment variables before sharing or pushing this repository.

## Data Files

The app auto-creates missing CSV/XLSX data files on startup.

Generated runtime artifacts (virtualenv, caches, attendance output, and local images) are excluded via `.gitignore`.