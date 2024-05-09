
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
import random
import string
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('vader_lexicon')
st.title("Cyberbullying Detection App")
st.write("This application helps identify potential cyberbullying based on text analysis.")
st.header("Understanding Bullying:")

st.write("**Positive statements can be bullying:**  Praising someone excessively to make them feel bad about themselves is a form of bullying.")
st.write("**Negative statements aren't always bullying:**  Constructive criticism or pointing out a mistake isn't necessarily bullying.")



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

# Function to get sentiment of a text
def get_sentiment(text):
    compound_score = sia.polarity_scores(text)['compound']
    return 'positive' if compound_score >= 0 else 'negative'

def res(arr):
    #print(arr)
    if arr[0] > 0.5:
       return "Bullying"
    else:
       return "Not bullying"

def cate(val, df, dic):
    index = dic[val]
    return df["Label"][index]
"""
def print_my_results_(inputs, results):
  result_for_printing = \
    [f'input: {inputs[i]:<10} :Sentiment: {get_sentiment(inputs[i])} : category: {res(results[i])}'
                         for i in range(len(inputs))]
  print(*result_for_printing, sep='\n')
  print()
"""

def print_my_results(inputs, results):
  """Prints (or displays in Streamlit) analysis results for each input."""

  for i in range(len(inputs)):
    st.write(f'**Input:** {inputs[i][:30]}')
    st.write(f'**Sentiment:** {results[i]}')
    st.write("---")

def clean_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    
    # Remove mentions and hashtags
    tweet = re.sub(r'@[A-Za-z0-9_]+|#[A-Za-z0-9_]+', '', tweet)
    
    # Remove special characters, numbers, and punctuation
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)
    
    # Remove 'RT' (Retweet) indicator
    tweet = re.sub(r'\bRT\b', '', tweet)
    
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Remove stopwords
    # stop_words = set(stopwords.words('english'))
    # tweet_tokens = nltk.word_tokenize(tweet)
    # tweet = ' '.join([word for word in tweet_tokens if word not in stop_words])
    
    # Lemmatization
    doc = nlp(tweet)
    # Lemmatize each token and join them back into a string
    tweet = ' '.join([token.lemma_ for token in doc])
    
    return tweet
  
def generate_preset_inputs():
  # Assuming 'df' is DataFrame
  random_selections = {}
  # Loop to get 10 random selections with their indices
  for _ in range(10):
    # Get a random index
    random_index = random.randint(0, len(df)-1)
    # Extract the item
    random_item = df.loc[random_index, 'Text'] 
    # Add to dictionary with index as key and item as value
    random_selections[random_item] = random_index
    return random_selections
    
"""
def print_my_results(inputs, results):
  result_for_printing = [
      f'input: {inputs[i][:30]} :Sentiment: {get_sentiment(inputs[i])} :'
      f' Original category: {cate(inputs[i], df, random_selections)} :'
      f' category: {res(results[i])}'
      for i in range(len(inputs))
  ]

  # Use st.write to display the results in Streamlit
  st.write(*result_for_printing, sep='\n')
  st.write("")  # Add an empty line for better formatting
"""

model_path = os.path.dirname(__file__)
#model_url = "https://www.dropbox.com/scl/fi/3ifsodhw1dbo9kw8behl3/cyberbullying_dbert.zip?rlkey=g49hb39f8sc8j334wreqlonok&st=6v4iz3wn&dl=0" 
model_url = "https://www.dropbox.com/scl/fi/zxmmulmsidprp08c50vjv/Dbert2.zip?rlkey=mq48q8nnkvrxcexdjgqq1eyie&st=e52l7zo5&dl=1"

model_bytes, content = download_model(model_url)
#check
#check_model(content)
sia = SentimentIntensityAnalyzer()
df = pd.read_csv("sample_data.csv")

examples_index = list(random_selections.values())
examples_text = list(random_selections.keys())

# Extract the zipped model content
reloaded_model = extract_model(model_bytes)



#__Main program starts

st.header("Analyze Text")

user_input = st.text_input("Enter text to analyze:")

if st.button("Analyze"):
  # Pass user_input to your cyberbullying detection model (replace with your model)
  prepro_input = clean_tweet(user_input)
  prediction = tf.sigmoid(reloaded_model(tf.constant(prepro_input)))
  print_my_results([user_input], prediction)

if st.button("Analyze with Preset Inputs"):
  inputs = generate_preset_inputs()
  results = tf.sigmoid(reloaded_model(tf.constant(inputs)))
  print_my_results(inputs, results)
  

"""
try:
  #model = tf.saved_model.load(model_path)
  model_bytes, content = download_model(model_url)
  #check
  check_model(content)
  # Extract the zipped model content
  model_content = extract_model(model_bytes)
except Exception as error:
  st.write(f"error: {error}")
  
"""
