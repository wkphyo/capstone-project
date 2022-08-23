import pandas as pd
import numpy as np
import requests
import pickle
import streamlit as st
import re
import joblib

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


imdb_df = pd.read_csv('imdb-titles-df.csv')[['imdbId', 'Title']]
truncated_final_df = pd.read_csv('collab-model-final-df-processed.csv')
features_matrix = joblib.load('collab-model-features-matrix.pkl')
model = joblib.load('collab-model.pkl')

features_df = truncated_final_df.pivot(
    index='movieId',
    columns='userId',
    values='rating'
    ).fillna(0).reset_index()

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


def recommend(title):
    n_movies_to_recommend = 10
    movie_list = truncated_final_df[truncated_final_df['Title']==title]  
    if len(movie_list):
        found_title = movie_list.iloc[0]['Title']        
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = features_df[features_df['movieId']==movie_idx].index[0]

        distances, indices = model.kneighbors(features_matrix[movie_idx],n_neighbors=n_movies_to_recommend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        rec_movie_indices.reverse()
        
        res = []

        for val in rec_movie_indices:
          movie_idx = features_df.iloc[val[0]]['movieId']
          idx = truncated_final_df[truncated_final_df['movieId'] == movie_idx].iloc[0]['Title']
          res.append(
              {'Title':idx, 'Distance':val[1]}
          )

        res = pd.DataFrame(res)
        res = pd.merge(res, imdb_df, how='left', on='Title')
        title_list = res.head(5).Title.to_list()
        imdbId_list = res.head(5).imdbId.to_list()
        overview_list = [fetch_overview(i) for i in imdbId_list]
        poster_list = [fetch_poster(i) for i in imdbId_list]
        emotion_list = [predict_emotion(i) for i in overview_list]
        return title_list, overview_list, poster_list, emotion_list
    else:
        return "No movies found. Please try again with a different title."


st.header("Movie Recommendation System 🎦")
st.write('\n')
st.markdown(
    "__*Built by*__: _Win Phyo_")


movie_list = truncated_final_df.Title.unique()
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

button_text = 'Show me recommendations based on what other people are watching 😎'
if st.button(button_text):
    names, overviews, posters, emotions = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(names[0])
        try:
            st.image(posters[0])
        except:
            st.write('Image not found')
        st.write(overviews[0])
        st.write('Emotion: ' + emotions[0])
    with col2:
        st.text(names[1])
        try:
            st.image(posters[1])
        except:
            st.write('Image not found')
        st.write(overviews[1])
        st.write('Emotion: ' + emotions[1])
    with col3:
        st.text(names[2])
        try:
            st.image(posters[2])
        except:
            st.write('Image not found')
        st.write(overviews[2])
        st.write('Emotion: ' + emotions[2])
    with col4:
        st.text(names[3])
        try:
            st.image(posters[3])
        except:
            st.write('Image not found')
        st.write(overviews[3])
        st.write('Emotion: ' + emotions[3])
    with col5:
        st.text(names[4])
        try:
            st.image(posters[4])
        except:
            st.write('Image not found')
        st.write(overviews[4])
        st.write('Emotion: ' + emotions[4])