
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import shutil
import tensorflow_hub as hub
import tensorflow_text as text
import os
import requests 
from io import BytesIO

st.title("Cyber Bullying detection App")
st.write("Hi bully and the buliied")

# download model from Dropbox, cache it and load the model into the app 
@st.cache(allow_output_mutation=True)
def download_model(url):
  """Downloads a zipped model file from the specified URL using requests."""
  model_response = requests.get(url)
  model_response.raise_for_status()  # Raise error for failed downloads
  return model_response.content

def extract_model(model_bytes):
  """Extracts the zipped model content into a BytesIO object representing an in-memory folder."""
  import zipfile
  model_zip = zipfile.ZipFile(BytesIO(model_bytes), 'r')
  model_content = {}
  for filename in model_zip.namelist():
    file_content = model_zip.read(filename)
    model_content[filename] = BytesIO(file_content)
  return model_content

def load_bert_model(model_content):
  """Loads the BERT model from the extracted files in the model_content dictionary."""
  from transformers import TFBertModel
  # Identify the file path for the saved model (assuming it's consistent)
  model_path = [path for path in model_content.keys() if 'saved_model.pb' in path][0]
  reloaded_model = tf.saved_model.load(BytesIO(model_content[model_path]))
  return reloaded_model

model_path = os.path.dirname(__file__)
model_url = "https://www.dropbox.com/scl/fi/3ifsodhw1dbo9kw8behl3/cyberbullying_dbert.zip?rlkey=g49hb39f8sc8j334wreqlonok&st=6v4iz3wn&dl=0" 
try:
  #model = tf.saved_model.load(model_path)
  model_bytes = download_model(model_url)
  # Extract the zipped model content
  model_content = extract_model(model_bytes)
except Exception as error:
  st.write(f"error: {error}")
  
