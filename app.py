##############################################################
# SMART FACE ATTENDANCE SYSTEM
# ArcFace ONNX + Mediapipe + Class/Subject Attendance
# Students only belong to CLASS (no subject in students.csv)
# Manual Mark Present + Send Email below it
##############################################################

import os
import time
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import mediapipe as mp
from PIL import Image
import streamlit as st

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


##############################################################
# CONFIG
##############################################################

IMAGES_DIR = "images"
STUDENT_CSV = "students.csv"
ATTENDANCE_FILE = "attendance.xlsx"
CLASSES_CSV = "classes.csv"
SUBJECTS_CSV = "subjects.csv"

ARC_MODEL_PATH = "models/w600k_r50.onnx"

SIM_THRESHOLD = 0.55         # cosine similarity threshold
TIME_REQUIRED = 2.0          # seconds same student must be visible
FACE_MIN_SIZE = 70           # minimum face width/height in px

ATTENDANCE_COLS = ["Roll", "Name", "Class", "Subject", "Time", "Date", "Status"]

# Email (for sending absentee emails)
SENDER_EMAIL = "yeshruthagowda@gmail.com"
SENDER_PASSWORD = "bnzd ybvj qwbo mcqe"    # Gmail App Password


##############################################################
# STREAMLIT APP CONFIG
##############################################################

os.makedirs(IMAGES_DIR, exist_ok=True)
st.set_page_config(page_title="Smart Attendance (GPU + Dashboard)", layout="wide")

# Initialize attendance file if missing
if not os.path.exists(ATTENDANCE_FILE):
    df_init = pd.DataFrame(columns=ATTENDANCE_COLS)
    df_init.to_excel(ATTENDANCE_FILE, index=False)


##############################################################
# GENERIC HELPERS
##############################################################

def load_attendance():
    """
    Safely load attendance.xlsx and make sure it has
    the columns: Roll, Name, Class, Subject, Time, Date, Status
    """
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=ATTENDANCE_COLS)
        df.to_excel(ATTENDANCE_FILE, index=False)
        return df

    df = pd.read_excel(ATTENDANCE_FILE)
    changed = False

    for col in ATTENDANCE_COLS:
        if col not in df.columns:
            df[col] = ""
            changed = True

    df = df[ATTENDANCE_COLS]

    if changed:
        try:
            df.to_excel(ATTENDANCE_FILE, index=False)
        except PermissionError:
            st.warning(
                "⚠️ Could not update 'attendance.xlsx' (file may be open). "
                "Close it in Excel and restart the app."
            )

    return df


##############################################################
# EMAIL UTIL
##############################################################

def send_absent_email(student_name, student_roll, student_class, subject, receiver_email):
    """Send absentee email to student's email address and show result in Streamlit."""
    subject_line = f"Absentee Alert - {student_name} ({student_roll})"

    body = f"""
Hello,

This is an automated attendance alert.

Student Name : {student_name}
Roll Number  : {student_roll}
Class        : {student_class}
Subject      : {subject}
Date         : {datetime.now().strftime('%Y-%m-%d')}

Status: ❌ ABSENT

Regards,
Smart Attendance System
"""

    msg = MIMEMultipart()
    msg["From"] = f"Smart Attendance App <{SENDER_EMAIL}>"
    msg["To"] = receiver_email
    msg["Subject"] = subject_line
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()

        st.write(f"🔒 Connecting to Gmail SMTP for: {receiver_email}")
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        st.write(f"✔ Logged in to Gmail as: {SENDER_EMAIL}")

        server.send_message(msg)
        server.quit()

        st.success(f"📨 Email sent to {receiver_email} for {student_name} ({student_roll})")
        return True

    except Exception as e:
        st.error(f"❌ Error sending email to {receiver_email}: {e}")
        return False


def send_all_absent_emails_for_class_subject(cls: str, subject: str):
    """Send absentee emails for a given class + subject for today's date."""
    students = load_students()
    abs_list = get_absentees(cls, subject)

    if not abs_list:
        st.success("✅ No absentees today for this class & subject.")
        return

    st.write(f"📧 Preparing to send emails for Class={cls}, Subject={subject}")
    success_count = 0
    fail_count = 0

    for a in abs_list:
        row = students[students["Roll"].astype(str) == a["Roll"]]
        if row.empty:
            st.warning(f"⚠ No student data found for roll {a['Roll']}")
            continue

        email = row.iloc[0].get("Email", "")
        if not email or "@" not in email:
            st.warning(f"⚠ No valid email for {a['Name']} (Roll {a['Roll']})")
            fail_count += 1
            continue

        ok = send_absent_email(a["Name"], a["Roll"], cls, subject, email)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    st.info(f"📊 Email sending done. ✅ Sent: {success_count} | ❌ Failed/Skipped: {fail_count}")


##############################################################
# ONNX ARC FACE MODEL
##############################################################

@st.cache_resource
def load_arcface_session():
    """Load ArcFace ONNX model with GPU/CPU providers."""
    if not os.path.exists(ARC_MODEL_PATH):
        raise FileNotFoundError(
            f"ArcFace model not found at {ARC_MODEL_PATH}. "
            "Place w600k_r50.onnx inside models/ folder."
        )
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(ARC_MODEL_PATH, providers=providers)
    input_name = sess.get_inputs()[0].name
    return sess, input_name


def preprocess_face_bgr(face_bgr: np.ndarray):
    """
    Safe preprocess:
    - Return None if face is invalid / too small / empty.
    - Otherwise return NCHW tensor for ArcFace.
    """
    if face_bgr is None:
        return None
    if face_bgr.size == 0:
        return None
    if len(face_bgr.shape) != 3:
        return None

    h, w, _ = face_bgr.shape
    if h < 20 or w < 20:
        return None

    face = cv2.resize(face_bgr, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32")
    face = (face - 127.5) / 128.0
    face = np.transpose(face, (2, 0, 1))  # HWC -> CHW
    face = np.expand_dims(face, axis=0)   # NCHW
    return face


def get_embedding_from_face(face_bgr: np.ndarray, sess, input_name: str):
    """Return L2-normalized 512D embedding, or None if invalid."""
    inp = preprocess_face_bgr(face_bgr)
    if inp is None:
        return None
    try:
        emb = sess.run(None, {input_name: inp})[0].flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb
    except Exception as e:
        print("Embedding error:", e)
        return None


##############################################################
# CLASSES & SUBJECTS HELPERS
##############################################################

@st.cache_data
def load_classes():
    if not os.path.exists(CLASSES_CSV):
        df = pd.DataFrame(columns=["Class"])
        df.to_csv(CLASSES_CSV, index=False)
        return df
    return pd.read_csv(CLASSES_CSV)


def save_classes(df):
    df.to_csv(CLASSES_CSV, index=False)
    load_classes.clear()


@st.cache_data
def load_subjects():
    if not os.path.exists(SUBJECTS_CSV):
        df = pd.DataFrame(columns=["Subject"])
        df.to_csv(SUBJECTS_CSV, index=False)
        return df
    return pd.read_csv(SUBJECTS_CSV)


def save_subjects(df):
    df.to_csv(SUBJECTS_CSV, index=False)
    load_subjects.clear()


##############################################################
# STUDENT DATA HELPERS (NO SUBJECT IN STUDENT TABLE)
##############################################################

@st.cache_data
def load_students():
    """
    Load students list and ensure required columns:
    Roll, Name, Class, Image, Email
    Drop old 'Subject' column if exists.
    """
    required_cols = ["Roll", "Name", "Class", "Image", "Email"]

    if not os.path.exists(STUDENT_CSV):
        df = pd.DataFrame(columns=required_cols)
        try:
            df.to_csv(STUDENT_CSV, index=False)
        except PermissionError:
            st.warning(
                "⚠️ Cannot create 'students.csv'. Close it in Excel/other apps and restart."
            )
        return df

    df = pd.read_csv(STUDENT_CSV)
    changed = False

    if "Subject" in df.columns:
        df = df.drop(columns=["Subject"])
        changed = True

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
            changed = True

    df = df[required_cols]

    if changed:
        try:
            df.to_csv(STUDENT_CSV, index=False)
        except PermissionError:
            st.warning(
                "⚠️ 'students.csv' is open in another program. "
                "Close it so the app can update its format."
            )

    return df


def save_students_df(df: pd.DataFrame):
    df.to_csv(STUDENT_CSV, index=False)
    load_students.clear()


def save_student(roll: str, name: str, cls: str, img_path: str, email: str):
    df = load_students().copy()
    df.loc[len(df)] = [roll, name, cls, img_path, email]
    save_students_df(df)


##############################################################
# ATTENDANCE HELPERS (CLASS + SUBJECT)
##############################################################

def clear_today_attendance_for_class_subject(cls: str, subject: str):
    df = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")
    df = df[~((df["Class"] == cls) & (df["Subject"] == subject) & (df["Date"] == today))]
    df.to_excel(ATTENDANCE_FILE, index=False)


def mark_attendance(roll: str, name: str, cls: str, subject: str):
    df = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")

    exists = (
        (df["Roll"].astype(str) == str(roll))
        & (df["Class"] == cls)
        & (df["Subject"] == subject)
        & (df["Date"] == today)
    ).any()

    if exists:
        return

    now = datetime.now().strftime("%H:%M:%S")
    df.loc[len(df)] = [roll, name, cls, subject, now, today, "Present"]
    df.to_excel(ATTENDANCE_FILE, index=False)
    st.toast(f"🟢 Marked Present: {name} ({roll}) – {cls} / {subject}", icon="🎉")


def manual_mark_present(roll: str, name: str, cls: str, subject: str):
    """Manual mark present – used in absentees section."""
    df = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")

    exists = (
        (df["Roll"].astype(str) == str(roll))
        & (df["Class"] == cls)
        & (df["Subject"] == subject)
        & (df["Date"] == today)
    ).any()

    if exists:
        return False

    now = datetime.now().strftime("%H:%M:%S")
    df.loc[len(df)] = [roll, name, cls, subject, now, today, "Present"]
    df.to_excel(ATTENDANCE_FILE, index=False)
    return True


def get_absentees(cls: str, subject: str):
    """
    Absentees = all students in class - those marked present in (class, subject, today)
    """
    students = load_students()
    cls_students = students[students["Class"] == cls]

    if cls_students.empty:
        return []

    df = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")
    df_today = df[(df["Class"] == cls) & (df["Subject"] == subject) & (df["Date"] == today)]

    present_rolls = set(df_today["Roll"].astype(str))
    all_rolls = set(cls_students["Roll"].astype(str))

    abs_rolls = sorted(list(all_rolls - present_rolls))
    result = []
    for r in abs_rolls:
        row = cls_students[cls_students["Roll"].astype(str) == r].iloc[0]
        result.append({
            "Roll": r,
            "Name": row["Name"],
            "Class": row["Class"],
            "Subject": subject
        })
    return result


##############################################################
# BUILD EMBEDDINGS FOR A CLASS
##############################################################

def build_embeddings_for_class(class_name: str):
    students = load_students()
    cls_students = students[students["Class"] == class_name]

    if cls_students.empty:
        return []

    sess, input_name = load_arcface_session()
    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )

    db = []
    st.info(f"Preparing embeddings for class: {class_name}...")

    for _, row in cls_students.iterrows():
        img_path = row["Image"]
        img = cv2.imread(img_path)
        if img is None:
            st.warning(f"Cannot load image for {row['Name']} -> {img_path}")
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = mp_face.process(rgb)

        if not dets.detections:
            st.warning(f"No face detected for {row['Name']} in {img_path}")
            continue

        h, w, _ = img.shape
        best = None
        best_area = 0

        for det in dets.detections:
            box = det.location_data.relative_bounding_box
            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)
            width = int(box.width * w)
            height = int(box.height * h)
            x2 = x1 + width
            y2 = y1 + height
            area = width * height
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)

        if best is None:
            continue

        x1, y1, x2, y2 = best
        # clamp box
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        face_crop = img[y1:y2, x1:x2]
        emb = get_embedding_from_face(face_crop, sess, input_name)
        if emb is None:
            continue

        db.append({
            "roll": str(row["Roll"]),
            "name": row["Name"],
            "class": row["Class"],
            "emb": emb
        })

        st.write(f"✔ Embedding ready: {row['Name']}")

    return db


##############################################################
# ATTENDANCE SESSION (AUTO MARKING) - CLASS + SUBJECT
##############################################################

def run_attendance_for_class_subject(class_name: str, subject: str, duration_minutes: int):
    st.subheader(f"📸 Attendance Session — Class: {class_name} | Subject: {subject}")

    clear_today_attendance_for_class_subject(class_name, subject)
    db = build_embeddings_for_class(class_name)
    if not db:
        st.error("No valid student embeddings for this class.")
        return

    sess, input_name = load_arcface_session()
    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    frame_display = st.empty()
    info = st.empty()

    info.info(
        f"Camera ON for {duration_minutes} minutes.\n"
        f"Class = {class_name}, Subject = {subject}\n"
        f"Similarity threshold = {SIM_THRESHOLD}, "
        f"Hold face steady for {TIME_REQUIRED} seconds."
    )

    start_session = time.time()
    last_candidate = None
    start_time_candidate = None
    elapsed = 0.0

    while True:
        if time.time() - start_session > duration_minutes * 60:
            st.success("⏱ Session finished.")
            break

        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from camera.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = mp_face.process(rgb)

        if not dets.detections:
            last_candidate = None
            start_time_candidate = None
            elapsed = 0.0
            frame_display.image(rgb)
            continue

        h, w, _ = frame.shape
        best_det = None
        best_area = 0

        for det in dets.detections:
            box = det.location_data.relative_bounding_box
            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)
            width = int(box.width * w)
            height = int(box.height * h)
            x2 = x1 + width
            y2 = y1 + height
            area = width * height

            if area > best_area and width >= FACE_MIN_SIZE and height >= FACE_MIN_SIZE:
                best_area = area
                best_det = (x1, y1, x2, y2)

        if best_det is None:
            last_candidate = None
            start_time_candidate = None
            elapsed = 0.0
            frame_display.image(rgb)
            continue

        x1, y1, x2, y2 = best_det
        # clamp box
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            last_candidate = None
            start_time_candidate = None
            elapsed = 0.0
            frame_display.image(rgb)
            continue

        face_crop = frame[y1:y2, x1:x2]
        emb_live = get_embedding_from_face(face_crop, sess, input_name)
        if emb_live is None:
            last_candidate = None
            start_time_candidate = None
            elapsed = 0.0
            frame_display.image(rgb)
            continue

        sims = [float(np.dot(emb_live, s["emb"])) for s in db]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        stu = db[best_idx]

        color = (0, 255, 0) if best_sim >= SIM_THRESHOLD else (0, 0, 255)
        label = f"{stu['name']} sim={best_sim:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        now = time.time()

        if best_sim >= SIM_THRESHOLD:
            if last_candidate == stu["roll"]:
                elapsed = now - (start_time_candidate or now)
            else:
                last_candidate = stu["roll"]
                start_time_candidate = now
                elapsed = 0.0
        else:
            last_candidate = None
            start_time_candidate = None
            elapsed = 0.0

        cv2.putText(frame, f"time={elapsed:.2f}s",
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0), 2)

        if elapsed >= TIME_REQUIRED and last_candidate is not None:
            mark_attendance(stu["roll"], stu["name"], stu["class"], subject)
            last_candidate = None
            start_time_candidate = None
            elapsed = 0.0

        frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    # ✅ mark in session_state that attendance session completed for this class+subject
    st.session_state["last_attendance_class"] = class_name
    st.session_state["last_attendance_subject"] = subject
    st.session_state["last_attendance_done"] = True

    st.info("✅ Attendance session complete. Scroll down to review absentees, mark present, or send emails.")


##############################################################
# REGISTRATION PAGE (CLASS ONLY)
##############################################################

def registration_page():
    st.header("🧑‍🎓 Register New Student")

    classes_df = load_classes()
    class_options = sorted(classes_df["Class"].dropna().unique().tolist())

    if not class_options:
        st.error("No classes found. Please add classes in Management → Class Management first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        roll = st.text_input("Roll Number")
        name = st.text_input("Full Name")
        cls = st.selectbox("Class", class_options)
        email = st.text_input("Student Email (for absent alert)")
    with col2:
        img = st.camera_input("Capture Face")

    if st.button("Register Student"):
        if not roll or not name or not cls or not email:
            st.error("Please fill Roll, Name, Class and Email.")
            return
        if img is None:
            st.error("Please capture the student's face.")
            return

        safe_name = name.replace(" ", "_")
        img_path = os.path.join(IMAGES_DIR, f"{roll}_{safe_name}.jpg")

        frame = Image.open(img)
        frame_np = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, frame_np)

        save_student(roll, name, cls, img_path, email)
        st.success(f"✅ Registered: {name} ({roll}) – {cls}")


##############################################################
# DASHBOARD PAGE (VIEW ATTENDANCE)
##############################################################

def dashboard_page():
    st.header("📊 Attendance Dashboard")

    df = load_attendance()
    if df.empty:
        st.info("No attendance records yet.")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    df_today = df[df["Date"] == today]

    st.subheader(f"Today's Attendance — {today}")
    st.dataframe(df_today, use_container_width=True)

    st.download_button(
        "⬇️ Download Today's Attendance (CSV)",
        df_today.to_csv(index=False).encode("utf-8"),
        f"attendance_{today}.csv",
        "text/csv"
    )


##############################################################
# MANUAL ATTENDANCE PAGE (CLASS + SUBJECT)
##############################################################

def manual_attendance_page():
    st.header("🕒 Manual Attendance (Teacher Mode)")

    students = load_students()
    if students.empty:
        st.error("No students registered yet.")
        return

    class_options = sorted(students["Class"].dropna().unique().tolist())
    subjects_df = load_subjects()
    subject_options = sorted(subjects_df["Subject"].dropna().unique().tolist())

    if not class_options:
        st.error("No classes found. Add classes in Management first.")
        return
    if not subject_options:
        st.error("No subjects found. Add subjects in Management first.")
        return

    cls = st.selectbox("Select Class", class_options)
    subject = st.selectbox("Select Subject", subject_options)
    duration = st.number_input("Duration (minutes)", min_value=1, max_value=60, value=5)

    if st.button("Start Attendance Session"):
        # reset state for new session
        st.session_state["last_attendance_class"] = cls
        st.session_state["last_attendance_subject"] = subject
        st.session_state["last_attendance_done"] = False

        run_attendance_for_class_subject(cls, subject, duration)

    # ===== After camera session: show absentees + manual controls =====
    if (
        st.session_state.get("last_attendance_done", False)
        and st.session_state.get("last_attendance_class") == cls
        and st.session_state.get("last_attendance_subject") == subject
    ):
        st.subheader("📋 Absentees For This Session")

        abs_list = get_absentees(cls, subject)
        if abs_list:
            abs_df = pd.DataFrame(abs_list)
            st.dataframe(abs_df, use_container_width=True)

            st.download_button(
                "⬇️ Download Absentees CSV",
                abs_df.to_csv(index=False).encode("utf-8"),
                f"absentees_{cls}_{subject}_{datetime.now().strftime('%Y-%m-%d')}.csv",
                "text/csv"
            )

            st.write("### ➕ Manually Mark Attendance")
            for a in abs_list:
                c1, c2, c3 = st.columns([2, 4, 3])
                with c1:
                    st.write(f"**{a['Roll']}**")
                with c2:
                    st.write(a["Name"])
                with c3:
                    if st.button(f"Mark Present ({a['Roll']})", key=f"mp_{a['Roll']}"):
                        ok = manual_mark_present(a["Roll"], a["Name"], a["Class"], subject)
                        if ok:
                            st.success(f"✔ {a['Name']} marked present manually.")
                        else:
                            st.info(f"{a['Name']} already present.")
                        st.rerun()

            st.write("---")
            if st.button("📧 Send Absentee Emails (Remaining Absentees)"):
                send_all_absent_emails_for_class_subject(cls, subject)

        else:
            st.success("🎉 All students are present for this class & subject.")


##############################################################
# MANAGEMENT DASHBOARD (ADVANCED)
##############################################################

def management_dashboard():
    st.header("🛠️ Management Dashboard")

    students = load_students()
    classes_df = load_classes()
    subjects_df = load_subjects()

    # ================== STATS ================== #
    st.subheader("📈 Summary")
    total_students = len(students)
    total_classes = classes_df["Class"].nunique()
    total_subjects = subjects_df["Subject"].nunique()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Students", total_students)
    c2.metric("Total Classes", total_classes)
    c3.metric("Total Subjects", total_subjects)

    st.write("---")

    # ================== CLASS MANAGEMENT ================== #
    st.subheader("📚 Class Management")

    existing_classes = sorted(classes_df["Class"].dropna().unique().tolist())
    st.write("Existing Classes List:")
    st.dataframe(classes_df, use_container_width=True)

    new_class = st.text_input("Add New Class")
    if st.button("Add Class"):
        if new_class:
            if new_class not in existing_classes:
                classes_df_new = classes_df.copy()
                classes_df_new.loc[len(classes_df_new)] = [new_class]
                save_classes(classes_df_new)
                st.success(f"Class '{new_class}' added.")
            else:
                st.info("Class already exists.")
        else:
            st.error("Class name cannot be empty.")

    del_class = st.selectbox("Delete Class from List (does NOT delete students)", ["<None>"] + existing_classes)
    if st.button("Delete Class from List"):
        if del_class != "<None>":
            classes_df_new = classes_df[classes_df["Class"] != del_class]
            save_classes(classes_df_new)
            st.success(f"Class '{del_class}' removed from list (students remain).")

    st.write("---")

    # ================== SUBJECT MANAGEMENT ================== #
    st.subheader("📘 Subject Management")

    subj_list = sorted(subjects_df["Subject"].dropna().unique().tolist())
    st.write("Existing Subjects List:")
    st.dataframe(subjects_df, use_container_width=True)

    new_subject = st.text_input("Add New Subject")
    if st.button("Add Subject"):
        if new_subject:
            if new_subject not in subj_list:
                subjects_df_new = subjects_df.copy()
                subjects_df_new.loc[len(subjects_df_new)] = [new_subject]
                save_subjects(subjects_df_new)
                st.success(f"Subject '{new_subject}' added.")
            else:
                st.info("Subject already exists.")
        else:
            st.error("Subject name cannot be empty.")

    del_subject = st.selectbox("Delete Subject from List", ["<None>"] + subj_list)
    if st.button("Delete Subject from List"):
        if del_subject != "<None>":
            subjects_df_new = subjects_df[subjects_df["Subject"] != del_subject]
            save_subjects(subjects_df_new)
            st.success(f"Subject '{del_subject}' removed from list.")

    st.write("---")

    # ================== STUDENT MANAGEMENT ================== #
    st.subheader("👨‍🎓 Student Management")

    col1, col2 = st.columns(2)
    class_filter = col1.selectbox("Filter by Class", ["All"] + sorted(students["Class"].dropna().unique().tolist()))
    search = col2.text_input("Search by Name/Roll")

    df_filtered = students.copy()

    if class_filter != "All":
        df_filtered = df_filtered[df_filtered["Class"] == class_filter]
    if search:
        s = search.lower()
        df_filtered = df_filtered[
            df_filtered["Name"].str.lower().str.contains(s) |
            df_filtered["Roll"].astype(str).str.contains(s)
        ]

    st.write("### Current Students")
    st.dataframe(df_filtered, use_container_width=True)

    st.write("---")

    # ----- Edit Student ----- #
    st.subheader("✏️ Edit Student")

    edit_roll = st.text_input("Enter Roll Number to Edit")
    if st.button("Load Student Details"):
        if not edit_roll:
            st.error("Enter a roll number.")
        else:
            students = load_students()
            row = students[students["Roll"].astype(str) == edit_roll]
            if row.empty:
                st.error("No student found with that roll.")
            else:
                row = row.iloc[0]
                st.session_state["edit_roll"] = str(row["Roll"])
                st.session_state["edit_name"] = row["Name"]
                st.session_state["edit_class"] = row["Class"]
                st.session_state["edit_email"] = row["Email"]

    if "edit_roll" in st.session_state:
        st.write(f"Editing Roll: **{st.session_state['edit_roll']}**")

        e_name = st.text_input("Name", value=st.session_state["edit_name"])
        e_class = st.text_input("Class", value=st.session_state["edit_class"])
        e_email = st.text_input("Email", value=st.session_state["edit_email"])

        if st.button("Save Changes"):
            students = load_students()
            idx = students[students["Roll"].astype(str) == st.session_state["edit_roll"]].index
            if not idx.empty:
                i = idx[0]
                students.at[i, "Name"] = e_name
                students.at[i, "Class"] = e_class
                students.at[i, "Email"] = e_email
                save_students_df(students)
                st.success("Student details updated.")
                del st.session_state["edit_roll"]

    st.write("---")

    # ----- Delete Student ----- #
    st.subheader("🗑️ Delete Student")

    del_roll = st.text_input("Roll Number to Delete")
    if st.button("Delete Student"):
        if not del_roll:
            st.error("Enter a roll number.")
        else:
            students = load_students()
            before = len(students)
            students_new = students[students["Roll"].astype(str) != del_roll]
            if len(students_new) == before:
                st.info("No student found with that roll.")
            else:
                save_students_df(students_new)
                st.success(f"Student with roll {del_roll} deleted.")


##############################################################
# MAIN APP
##############################################################

def main():
    st.sidebar.title("Teacher Panel")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Dashboard",
            "Register Student",
            "Manual Attendance",
            "Management"
        ]
    )

    if page == "Dashboard":
        dashboard_page()
    elif page == "Register Student":
        registration_page()
    elif page == "Manual Attendance":
        manual_attendance_page()
    elif page == "Management":
        management_dashboard()


if __name__ == "__main__":
    main()
