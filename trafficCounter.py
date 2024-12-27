import streamlit as st
import cv2

# Titlul aplicației
st.title("Numărătoare de mașini cu YOLO11")

# Upload pentru încărcarea unui fișier video
uploaded_video = st.file_uploader("Încarcă un videoclip", type=["mp4", "avi", "mov"])

# Verifică dacă utilizatorul a încărcat un fișier
if uploaded_video:
    # Salvează videoclipul local pentru procesare
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    # Deschide videoclipul folosind OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Eroare la citirea videoclipului!")
    else:
        # Obține proprietățile videoclipului: lățime, înălțime și fps
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Definește o linie de interes pentru numărarea mașinilor
        line_points = [(20, 400), (1080, 400)]

        try:
            # Importă și configurează modelul YOLO pentru numărare
            from ultralytics import solutions
            counter = solutions.ObjectCounter(model="yolo11n.pt", region=line_points)

            # Inițializează contorul pentru mașini
            count = 0

            # Iterează prin fiecare cadru al videoclipului
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Procesează cadrul și numără mașinile
                frame = counter.count(frame)
                count += counter.total_count

            # Afișează rezultatul final utilizatorului
            st.success(f"Număr total mașini: {count}")
        except ImportError:
            # Afișează un mesaj de eroare dacă lipsește biblioteca necesară
            st.error("Biblioteca Ultralytics nu este instalată. Asigurați-vă că ați inclus-o în requirements.txt.")

        # Eliberează resursele utilizate de OpenCV
        cap.release()
