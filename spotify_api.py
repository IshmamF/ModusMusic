import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

class SpotifyClient:
    def __init__(self, client_id, client_secret):
        """
        Initializes the SpotifyClient with given client credentials.
        :param client_id: Spotify API Client ID
        :param client_secret: Spotify API Client Secret
        """
        self.sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id, client_secret))

    def get_track_id(self, song_name):
        """
        Searches for a track by name and returns its Spotify ID.
        :param song_name: The name of the song to search for
        :return: Spotify track ID or None if not found
        """
        results = self.sp.search(q=song_name, type='track', limit=1)
        if results['tracks']['items']:
            return results['tracks']['items'][0]['id']
        else:
            print(f"No results found for {song_name}")
            return None

    def get_track_preview_url(self, track_id):
        """
        Retrieves the 30-second preview URL for a given track ID.
        :param track_id: Spotify track ID
        :return: Preview URL or None if not available
        """
        track_info = self.sp.track(track_id)
        return track_info.get('preview_url')
