import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit.components.v1 as components

# Function to analyze sentiment of the user's input
def analyze_user_input(user_input, tokenizer, model):
    encoded_input = tokenizer(user_input, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = model(encoded_input)
    scores = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
    predicted_class_idx = tf.argmax(outputs.logits, axis=-1).numpy()[0]
    sentiment_label = model.config.id2label[predicted_class_idx]
    sentiment_score = scores[predicted_class_idx]
    return sentiment_label, sentiment_score

# Function to match songs from the dataset with the user's sentiment
def match_songs_with_sentiment(user_sentiment_label, user_sentiment_score,inputVector, score_range,songs_df):

    # Filter songs with the same sentiment label
    matched_songs = songs_df[songs_df['sentiment'] == user_sentiment_label]

    # Calculate the score range
    score_min = max(0, user_sentiment_score - score_range)
    score_max = min(1, user_sentiment_score + score_range)

    # Further filter songs whose scores fall within the specified range
    matched_songs = matched_songs[(matched_songs['score'] >= score_min) & (matched_songs['score'] <= score_max)]

    # Shuffle the matched songs to get a random order
    matched_songs = matched_songs.sample(frac=1).reset_index(drop=True)

    matched_songs['similarity'] = matched_songs['seq'].apply(lambda x: util.pytorch_cos_sim(sim_model.encode(x), inputVector))

    top_5 = matched_songs['similarity'].sort_values(ascending=False).head(5).reset_index(drop=True)

    # Select the top five songs and return
    return matched_songs.loc[top_5.index, ['song','artist','seq','similarity','sentiment','score']]

# Initialize the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = TFRobertaForSequenceClassification.from_pretrained('arpanghoshal/EmoRoBERTa')
sim_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

client_id = 'c34955a27b6447e3a1b92305d04bbbea'
client_secret = '1d197925c0654b5da80bd3cfa1f5afdd'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_track_id(song_name):
    # Search for the track ID using the song name
    results = sp.search(q=song_name, type='track', limit=1)
    if results['tracks']['items']:
        track_id = results['tracks']['items'][0]['id']
        return track_id
    else:
        print(f"No results found for {song_name}")
        return None

def get_track_preview_url(track_id):
    # Get the 30-second preview URL for the track
    track_info = sp.track(track_id)
    preview_url = track_info['preview_url']
    return preview_url

st.set_page_config(page_title="Music Mood Matcher", layout="centered")
st.markdown("""
    <style>
    .stApp {
        font-family: 'Circular Spotify Text', Arial, sans-serif;
        background-color: #000000;
        color: #fff;
    }
    div.stTextInput > label:first-child { color: #d9d8d0; }
    div.stButton > button:first-of-type {
        background-color: #1db954;
        color: white;
        border: none;
    }
    div.stExpander > expander {
            color:#1db954;
    }        
    div.stButton > button:first-of-type:hover {
        background-color: #17a145;
        color: white;
        border: none;
    }
    .stExpander {
        color: #1db954 !important;
    }
    .stExpander:hover {
        color: white !important;
    }
    .separator {
        height: 2px;
        background-color: #333;
        border: none;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


# Title and input section
st.markdown("<h1 style='text-align: center; color: #fff; font-family: 'Circular Spotify Text', Arial, sans-serif;'>Song Search</h1>", unsafe_allow_html=True)
user_input = st.text_input("Enter your text:", placeholder="Type your text here...", key="userInput")
submit_button = st.button("Generate", key="submit")

if submit_button and len(user_input.split()) > 5:
    inputVector = sim_model.encode(user_input)
    sentiment_label, sentiment_score = analyze_user_input(user_input, tokenizer, model)
    st.write(f"Sentiment: {sentiment_label}, Score: {sentiment_score:.2f}")
    songs_df = pd.read_csv('./music_mental_health.csv')
    suggested_songs = match_songs_with_sentiment(sentiment_label, sentiment_score, inputVector, 0.00625, songs_df)
    suggested_songs['similarity'] = suggested_songs['similarity'].apply(lambda x: x.numpy()[0][0])

    for index, row in suggested_songs.iterrows():
        track_id = get_track_id(row['song'])
        
        # Song header
        st.markdown(f"<div class='song-header'><h3 style='color: #00FF00'>{row['song']} - {row['artist']}</h3></div>", unsafe_allow_html=True)

        # Lyrics and audio within the song card
        with st.expander("Show Lyrics"):
            st.caption(row['seq'])

        if track_id and track_id.strip():
            preview_url = get_track_preview_url(track_id)
            if preview_url:
                st.audio(preview_url)
            else:
                st.write("No Preview Available")
                
        components.html("""<hr class="separator" /> """)

elif submit_button and not len(user_input.split()) > 5:
    st.warning("Please provide a longer response with 5 words or more.")
    st.rerun()