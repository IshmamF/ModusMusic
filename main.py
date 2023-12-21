import streamlit as st
from spotify_api import SpotifyClient
from sentiment_analysis import SentimentAnalyzer
from song_matching import SongMatcher

# Spotify API credentials - ensure these are securely stored or use environment variables
CLIENT_ID = "c34955a27b6447e3a1b92305d04bbbea"
CLIENT_SECRET = "1d197925c0654b5da80bd3cfa1f5afdd"

# Initialize SpotifyClient, SentimentAnalyzer, and SongMatcher
spotify_client = SpotifyClient(CLIENT_ID, CLIENT_SECRET)
sentiment_analyzer = SentimentAnalyzer()
song_matcher = SongMatcher('./music_mental_health.csv')

# Streamlit app layout
st.set_page_config(page_title="MODUS MUSIC", layout="wide")  # New: Setting page title and layout

# Custom CSS for background and text color
st.markdown("""
    <style>
    .stApp {
        background: rgb(0,0,0);
        background-size: cover;
        color: white;  /* Sets global text color to white */
    }
    /* General rule for all labels */
    label {
        color: white !important;
    }
    /* Specific color for the main title */
    h1 {
        color: red !important;  /* Making the MODUS MUSIC title red */
    }
    /* Additional specific styling */
    .stTextInput > label, .stButton > button, .css-10trblm, .css-1yjuwjr, .intro {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)


image_path = './MODUSMUSIC.png'  # Replace with the actual path to your image

st.image(image_path, use_column_width=False, width=250)  # Adjust the width as needed
# Custom gradient background using CSS
st.markdown("""
    <style>
    .stApp {
        background: rgb(0,0,0);
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

# Custom HTML for the main title
st.markdown("<h1 style='text-align: center; font-weight: bold;'>MODUS MUSIC</h1>", unsafe_allow_html=True)

st.title('Music Suggestion Based on Your Feeling')  # Existing Title

# New: Introduction Section
with st.container():
    st.markdown("""
        <style>
        .intro {
            font-size:18px;
        }
        </style>
        <div class='intro'>
        Welcome to Modus Music! Share your vibe, and let's find the perfect songs to match your mood.
        Just type in your thoughts, and we'll do the rest.
        </div>
        """, unsafe_allow_html=True)

# User input text area
with st.container():
    user_input = st.text_area("What's your vibe? Tell me about it:", key="123", height=150, max_chars=500)
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(204, 49, 49);

}
</style>""", unsafe_allow_html=True)
# Use the custom style for the button
submit_button = st.button("Generate music")

# Processing and Displaying Results
if submit_button and user_input.strip():
    
    # Run sentiment analysis on the user input
    sentiment_label, sentiment_score = sentiment_analyzer.analyze_sentiment(user_input)
    st.write(f"Sentiment: {sentiment_label}, Score: {sentiment_score:.2f}")

    suggested_songs = song_matcher.match_songs_with_sentiment(sentiment_label, sentiment_score, user_input, 0.00625)

    with st.container():
        st.markdown("<div class='song-list'>", unsafe_allow_html=True)
        st.write("Based on your vibe, you might like these songs:")
        for index, row in suggested_songs.iterrows():
            song = row['song']
            artist = row['artist']
            track_id = spotify_client.get_track_id(song)
            if track_id:
                preview_url = spotify_client.get_track_preview_url(track_id)
                st.write(f"{song} by {artist}")
                with st.expander(f"Show Lyrics for {song} by {artist}", expanded=False):
                    st.write(f"Lyrics: {row['seq']}")
                if preview_url:
                    st.audio(preview_url)
                else:
                    st.write("No Preview Available")
            else:
                st.write(f"Unable to find {song} by {artist} on Spotify.")

        st.markdown("</div>", unsafe_allow_html=True)
elif submit_button and not user_input.strip():
    st.warning("Please provide a longer response with 5 words or more.")
    st.rerun()
