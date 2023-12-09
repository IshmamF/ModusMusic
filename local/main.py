import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sim_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

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

    top_5 = matched_songs['similarity'].sort_values(ascending=False).head(5)

    # Sort the songs by how close their score is to the user's sentiment score
    # matched_songs['score_diff'] = abs(matched_songs['score'] - user_sentiment_score)
    # matched_songs = matched_songs.sort_values(by='score_diff')

    # Select the top five songs and return
    return matched_songs.loc[top_5.index, ['song','artist','seq','similarity','sentiment','score']]

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

# Initialize the tokenizer and model outside of the functions to speed up repeated calls
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = TFRobertaForSequenceClassification.from_pretrained('arpanghoshal/EmoRoBERTa')

# Streamlit app layout
st.set_page_config(page_title="Music Mood Matcher", layout="wide")  # New: Setting page title and layout

# Custom gradient background using CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #6e8efb, #a777e3);  # Example gradient colors
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('Music Suggestion Based on Your Mood')  # Existing Title

# New: Introduction Section
with st.container():
    st.markdown("""
        <style>
        .intro {
            font-size:18px;
        }
        </style>
        <div class='intro'>
        Welcome to the Music Mood Matcher! Share how your day went, and let's find the perfect songs to match your mood.
        Just type in your thoughts, and we'll do the rest.
        </div>
        """, unsafe_allow_html=True)

# User input text area
with st.container():
    user_input = st.text_area("How was your day? Tell me about it:", key="123", height=150, max_chars=500)
    submit_button = st.button("Generate music")

# Processing and Displaying Results
if submit_button and len(user_input.split()) > 5:
    # New: Define inputVector here
    inputVector = sim_model.encode(user_input)

    # Run sentiment analysis on the user input
    sentiment_label, sentiment_score = analyze_user_input(user_input, tokenizer, model)
    st.write(f"Sentiment: {sentiment_label}, Score: {sentiment_score:.2f}")

    # Load songs dataframe
    songs_df = pd.read_csv('./music_mental_health.csv')

    # Suggest songs
    suggested_songs = match_songs_with_sentiment(sentiment_label, sentiment_score, inputVector, 0.00625, songs_df)
    suggested_songs['similarity'] = suggested_songs['similarity'].apply(lambda x: x.numpy()[0][0])

    # Styling for the suggested songs display
    with st.container():
        st.markdown("<div class='song-list'>", unsafe_allow_html=True)
        st.write("Based on your mood, you might like these songs:")
        for index, row in suggested_songs.iterrows():
          song = row['song']
          artist = row['artist']
          track_id = get_track_id(song)
          if track_id.strip():
            preview_url = get_track_preview_url(track_id)
            st.write(f"Similarity: {row['similarity']}")
            st.write(f"{song} by {artist}")
            st.write(f"Lyrics: {row['seq']}")
            if preview_url:
              st.audio(preview_url)
            else:
              st.write("No Preview Available")
        st.markdown("</div>", unsafe_allow_html=True)
        st.dataframe(suggested_songs[['song','artist','seq','similarity','sentiment','score']])
elif submit_button and not len(user_input.split()) > 5:
    st.warning("Please provide a longer response with 5 words or more.")
    st.rerun()
