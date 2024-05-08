
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

import tempfile
import zipfile

import string
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('vader_lexicon')

st.title("Cyber Bullying detection App")
st.write("Hi bully and the buliied")

# download model from Dropbox, cache it and load the model into the app 
@st.cache(allow_output_mutation=True)
def download_model(url):
  """Downloads a zipped model file from the specified URL using requests."""
  model_response = requests.get(url)
  model_response.raise_for_status()  # Raise error for failed downloads
  return model_response.content, model_response


def download_model(url):
  """Downloads a zipped model file from the specified URL using requests."""
  model_response = requests.get(url)
  model_response.raise_for_status()  # Raise error for failed downloads
  return model_response.content, model_response

def check_model(content):
  try:
    model_response = content
    #model_response.raise_for_status()  # Raise error for failed downloads

    # Extract file extension (assuming content-type header is present)
    content_type = model_response.headers.get('Content-Type')
    if content_type:
      file_extension = content_type.split('/')[-1]
    else:
      file_extension = ""
    print(file_extension)
    return True, file_extension
  except requests.exceptions.RequestException as e:
    print(f"Download failed: {e}")
    return False, ""

def extract_model(model_bytes):
  """Extracts the zipped model content into a BytesIO object representing an in-memory folder."""
  import zipfile
  # Extract the zipped model to a temporary directory (replace with your desired location)
  with zipfile.ZipFile(BytesIO(model_bytes), 'r') as zip_ref:
    zip_ref.extractall("extracted_model")
  # Load the model from the extracted directory
  loaded_model = tf.saved_model.load("extracted_model")
  return loaded_model


def extract_model_(model_bytes):
  """Extracts the zipped model content to a temporary location and prints the file paths.

  Returns:
      A list of extracted file paths.
  """
  # Create a temporary directory for extracted files
  with tempfile.TemporaryDirectory() as temp_dir:
    model_zip = zipfile.ZipFile(BytesIO(model_bytes), 'r')
    extracted_files = []
    for filename in model_zip.namelist():
      # Extract file to temporary directory
      file_path = os.path.join(temp_dir, filename)
      with open(file_path, 'wb') as outfile:
        outfile.write(model_zip.read(filename))
      extracted_files.append(file_path)
      print(f"Extracted file: {file_path}")
    return extracted_files

def load_bert_model(model_content):
  """Loads the BERT model from the extracted files in the model_content dictionary."""
  from transformers import TFBertModel
  print(model_content.keys())
  # Identify the file path for the saved model (assuming it's consistent)
  model_path = [path for path in model_content.keys() if 'saved_model.pb' in path][0]
  model_bytes = model_conten8t[model_path].read()
  # Extract the file path from the BytesIO object
  #model_file_path = model_content[model_path].name
  reloaded_model = tf.saved_model.load(BytesIO(model_bytes))
  #print(model_bytes)
  #reloaded_model = tf.saved_model.load(model_file_path)
  return reloaded_model

model_path = os.path.dirname(__file__)
model_url = "https://www.dropbox.com/scl/fi/3ifsodhw1dbo9kw8behl3/cyberbullying_dbert.zip?rlkey=g49hb39f8sc8j334wreqlonok&st=6v4iz3wn&dl=0" 
try:
  #model = tf.saved_model.load(model_path)
  model_bytes, content = download_model(model_url)
  #check
  check_model(content)
  # Extract the zipped model content
  model_content = extract_model(model_bytes)
except Exception as error:
  st.write(f"error: {error}")
  
