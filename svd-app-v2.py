
import pandas as pd
import numpy as np
import requests
import pickle
import streamlit as st
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional,Dropout


st.set_page_config(page_title="Movie Recommendation System", page_icon="🎦")


final_res = pd.read_csv('svd-model-predictions-df.csv')[[
    'uid', 
    'Title', 
    'predictedRating',
    'imdbId'
    ]]

tokenizer = pickle.load(open("er-tokenizer.pickle", "rb"))
le = pickle.load(open("er-labelEncoder.pickle", "rb"))
emotion_model = keras.models.load_model('emotion-recognition-model')

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

def clean_new(sentence):
    """Pre-processing sentence for prediction"""
    stemmer = PorterStemmer()
    corpus = []
    text = re.sub("[^a-zA-Z]", " ", sentence)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    text = " ".join(text)
    corpus.append(text)
    one_hot_word = [one_hot(input_text=word, n=15092) for word in corpus]
    pad = pad_sequences(sequences=one_hot_word,maxlen=300,padding='pre')
    return pad


def predict_emotion(text):
    if text == 'Not found':
        result = 'Not found'
    else:
        text = clean_new(text)
        result = le.inverse_transform(
            np.argmax(emotion_model.predict(text), axis=-1)
            )[0]
    return result


def fetch_poster(imdbId):
    try:
        url = "https://api.themoviedb.org/3/find/{}?api_key=85fc3c49845e51a05cbaadc87fa820a8&language=en-US&external_source=imdb_id".format(imdbId)
        data = requests.get(url)
        data = data.json()
        poster_path = data['movie_results'][0]['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    except: 
        return 'Not found'


def fetch_overview(imdbId):
    try:
        url = "https://api.themoviedb.org/3/find/{}?api_key=85fc3c49845e51a05cbaadc87fa820a8&language=en-US&external_source=imdb_id".format(imdbId)
        data = requests.get(url)
        data = data.json()
        overview = data['movie_results'][0]['overview']
        return overview
    except: 
        return 'Not found'


def svd_recommend(uid):
    df = final_res[final_res['uid']==uid]
    title_list = df.Title.to_list()
    imdbId_list = df.imdbId.to_list()
    poster_path_list = [fetch_poster(i) for i in imdbId_list]
    overview_list = [fetch_overview(i) for i in imdbId_list]
    emotion_list = [predict_emotion(i) for i in overview_list]
    return title_list, poster_path_list, overview_list, emotion_list


st.header("Movie Recommendation System 🎦")
st.write('\n')
st.markdown(
    "__*Built by*__: _Win Phyo_")

uid_list = final_res.uid.unique()[0:100]
selected_uid= st.selectbox(
    "Select a user to see their top 5 recommended movies",
    uid_list
)

if selected_uid:
    names, posters,  overviews, emotions = svd_recommend(selected_uid)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write(names[0])
        try:
            st.image(posters[0])
        except:
            st.write('Image not found')
        st.write(overviews[0])
        st.write('Emotion: ' + emotions[0])
    with col2:
        st.write(names[1])
        try:
            st.image(posters[1])
        except:
            st.write('Image not found')
        st.write(overviews[1])
        st.write('Emotion: ' + emotions[1])
    with col3:
        st.write(names[2])
        try:
            st.image(posters[2])
        except:
            st.write('Image not found')
        st.write(overviews[2])
        st.write('Emotion: ' + emotions[2])
    with col4:
        st.write(names[3])
        try:
            st.image(posters[3])
        except:
            st.write('Image not found')
        st.write(overviews[3])
        st.write('Emotion: ' + emotions[3])
    with col5:
        st.write(names[4])
        try:
            st.image(posters[4])
        except:
            st.write('Image not found')
        st.write(overviews[4])
        st.write('Emotion: ' + emotions[4])

