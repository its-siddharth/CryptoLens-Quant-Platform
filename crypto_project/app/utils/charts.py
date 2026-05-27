import streamlit as st
import os
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')

def display_image(filename, caption="", width=None):
    """Display a local image from the data directory."""
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        img = Image.open(path)
        if width:
            st.image(img, caption=caption, width=width)
        else:
            st.image(img, caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {filename}")