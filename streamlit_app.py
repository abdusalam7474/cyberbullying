import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import shutil
import tensorflow_hub as hub
import tensorflow_text as text

st.title("Cyber Bullying detection App")
st.write("Hi bully and the buliied")
model_path = "/"
try:
  model = tf.saved_model.load(model_path)
except error:
  st.write(f"error: {error}")
