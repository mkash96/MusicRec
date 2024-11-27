from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

cache = {}

def get_youtube_thumbnail(song_name):
    if song_name in cache:
        return cache[song_name]

    try:
        youtube = build('youtube', 'v3', developerKey='AIzaSyDkYfXeLB8CZvXqV-e-Grlkscr5fp2uy0g')
        request = youtube.search().list(
            q=song_name,
            part='snippet',
            maxResults=1
        )
        response = request.execute()

        if response['items']:
            video_id = response['items'][0]['id']['videoId']
            thumbnail_url = response['items'][0]['snippet']['thumbnails']['high']['url']
            video_url = f'https://www.youtube.com/watch?v={video_id}'
            cache[song_name] = (thumbnail_url, video_url)
            return thumbnail_url, video_url
        else:
            return None, None
    except HttpError as e:
        if 'quotaExceeded' in str(e):
            print("YouTube API quota exceeded. Providing a search link.")
            search_url = f"https://www.youtube.com/results?search_query={song_name.replace(' ', '+')}"
            placeholder_thumbnail = 'https://via.placeholder.com/560x315?text=Quota+Exceeded'
            return placeholder_thumbnail, search_url
        raise
