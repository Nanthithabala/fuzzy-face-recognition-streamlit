import streamlit as st
import os
from train_profiles import train_profiles
from fuzzy_engine import fuzzy_match
from PIL import Image

RAW_DIR = "data/raw_images"

st.set_page_config(page_title="Fuzzy Face Recognition")
st.title("Fuzzy Face Recognition System")

menu = st.selectbox("Menu", ["Home", "Register Face", "Train Profiles", "Recognize Face"])

if menu == "Home":
    st.write("Fuzzy logic based face recognition demo")

elif menu == "Register Face":
    name = st.text_input("Person Name")
    images = st.file_uploader("Upload face images", accept_multiple_files=True)

    if st.button("Save"):
        if name and images:
            os.makedirs(os.path.join(RAW_DIR, name), exist_ok=True)
            for img in images:
                with open(os.path.join(RAW_DIR, name, img.name), "wb") as f:
                    f.write(img.getbuffer())
            st.success("Images saved")

elif menu == "Train Profiles":
    if st.button("Train"):
        train_profiles()
        st.success("Training completed")

elif menu == "Recognize Face":
    cam = st.camera_input("Capture Face")
    if cam:
        with open("temp.jpg", "wb") as f:
            f.write(cam.getbuffer())
        person, conf = fuzzy_match("temp.jpg")
        st.image(Image.open("temp.jpg"))
        st.success(f"Match: {person} ({conf}%)")
