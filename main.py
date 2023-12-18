
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, logging
import nltk
import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import streamlit as st
warnings.filterwarnings("ignore", category=DeprecationWarning)

st.header("----------HealthCare ChatBot----------")
if "messages" not in st.session_state: # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Tell me about your problems"}
    ]

@st.cache_data
def create_model():
  df = pd.read_excel('./Data1608.xlsx')
  # Get the number of elements in each class
  class_counts = df['LABEL'].value_counts()

  # Print the number of elements in each class
  for class_label, count in class_counts.items():
    print(f"{class_label}: {count} ")
  threshold = 60

  labels_to_remove = class_counts[class_counts < threshold].index
  df = df[~df['LABEL'].isin(labels_to_remove)]
  df = df.dropna()
  tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

  #X = df['Q'].apply(lambda text: tokenize(text, stopwords))
  X = df['Q1']
  y = df['LABEL'].values

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

  vectorizer = TfidfVectorizer()

  X_train_transformed = vectorizer.fit_transform(X_train)
  X_test_transformed = vectorizer.transform(X_test)

  svm = SVC()
  with open('./model.pkl', 'rb') as f:
    svm = pickle.load(f)
  return svm, vectorizer
  
model, vectorizer = create_model()


def infer(text, vectorizer, model):
    # Vectorize the text
    transformed_text = vectorizer.transform([text])

    # Make the prediction
    y_pred = model.predict(transformed_text)
    message = "Bạn nên thử khám chuyên khoa sau: "+ str(y_pred[0])
    return message

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = infer(str(prompt), vectorizer, model)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message) # Add response to message history
