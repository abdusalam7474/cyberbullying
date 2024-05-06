import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import shutil
import tensorflow_hub as hub
import tensorflow_text as text
import os

st.title("Cyber Bullying detection App")
st.write("Hi bully and the buliied")
model_path = os.path.dirname(__file__)
try:
  model = tf.saved_model.load(model_path)
except Exception as error:
  st.write(f"error: {error}")
