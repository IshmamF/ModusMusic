import pandas as pd
from sentence_transformers import SentenceTransformer, util

class SongMatcher:
    def __init__(self, songs_data_file, model_name="sentence-transformers/all-mpnet-base-v2"):
        """
        Initializes the SongMatcher with the songs data file and the SentenceTransformer model.
        :param songs_data_file: Path to the CSV file containing songs data
        :param model_name: Name of the SentenceTransformer model
        """
        self.songs_df = pd.read_csv(songs_data_file)
        self.sim_model = SentenceTransformer(model_name)

    def match_songs_with_sentiment(self, user_sentiment_label, user_sentiment_score, user_input, score_range):

        # New: Define inputVector here
        inputVector = self.sim_model.encode(user_input)
    
        # Filter songs with the same sentiment label
        matched_songs = self.songs_df[self.songs_df['sentiment'] == user_sentiment_label]
    
        # Calculate the score range
        score_min = max(0, user_sentiment_score - score_range)
        score_max = min(1, user_sentiment_score + score_range)
    
        # Further filter songs whose scores fall within the specified range
        matched_songs = matched_songs[(matched_songs['score'] >= score_min) & (matched_songs['score'] <= score_max)]
    
        # Shuffle the matched songs to get a random order
        matched_songs = matched_songs.sample(frac=1).reset_index(drop=True)
    
        matched_songs['similarity'] = matched_songs['seq'].apply(lambda x: util.pytorch_cos_sim(self.sim_model.encode(x), inputVector))
    
        top_5 = matched_songs['similarity'].sort_values(ascending=False).head(5)
    
        # Sort the songs by how close their score is to the user's sentiment score
        # matched_songs['score_diff'] = abs(matched_songs['score'] - user_sentiment_score)
        # matched_songs = matched_songs.sort_values(by='score_diff')
    
        # Select the top five songs and return
        return matched_songs.loc[top_5.index, ['song','artist','seq','similarity','sentiment','score']]
