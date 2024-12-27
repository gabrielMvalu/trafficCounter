import os
import streamlit as st
import subprocess

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

    try:
        # Extrage cadrele video folosind FFmpeg
        output_frames = "frames/output_frame_%04d.jpg"
        subprocess.run(["ffmpeg", "-i", video_path, output_frames], check=True)
        st.success("Cadrele video au fost extrase cu succes!")

        # Procesare YOLO pe cadrele extrase
        st.info("Urmează procesarea cadrelor cu YOLO...")
        # Implementarea YOLO pe cadre se face ulterior
    except Exception as e:
        st.error(f"A apărut o eroare la procesarea videoclipului: {e}")


