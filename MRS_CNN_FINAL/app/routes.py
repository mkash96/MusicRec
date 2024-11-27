from flask import Blueprint, jsonify, request, render_template
from .recommender import Recommender
from .utils.youtube_helper import get_youtube_thumbnail
import random

main = Blueprint('main', __name__)

# Initialize the Recommender class
recommender = Recommender(
    feature_vectors_path='Precomputed_Features/feature_vectors.npy',
    labels_path='Precomputed_Features/labels.npy'
)

@main.route('/')
def homepage():
    print("Rendering index.html...")
    return render_template('index.html')


@main.route('/get_songs', methods=['GET'])
def get_songs():
    return jsonify({"songs": recommender.get_song_list().tolist()})


@main.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    seed_song = data.get("seed_song")
    shown_songs = data.get("shown_songs", [])

    try:
        recommendations = recommender.recommend_songs(seed_song, top_n=2, exclude_songs=shown_songs)
        if not recommendations:
            return jsonify({"error": "No more recommendations available."}), 404
        response = [{"song": rec[0], "score": rec[1]} for rec in recommendations]
        return jsonify({"recommendations": response})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404



@main.route('/surprise_me', methods=['POST'])
def surprise_me():
    data = request.json
    seed_song = data.get("seed_song")
    first = data.get("first", False)

    try:
        # Get all recommendations
        recommendations = recommender.recommend_songs(seed_song, top_n=None)
        
        if first:
            # Get the least similar song
            least_similar_song = recommendations[-1]  # Least similar song
            response = {
                "song": least_similar_song[0],
                "score": least_similar_song[1],
            }
        else:
            # Get songs with similarity between 0 and 0.6
            surprise_songs = [rec for rec in recommendations if 0 <= rec[1] <= 0.6]
            if not surprise_songs:
                return jsonify({"error": "No surprise songs available."}), 404
            # Randomly select one
            surprise_song = random.choice(surprise_songs)
            response = {
                "song": surprise_song[0],
                "score": surprise_song[1],
            }
        return jsonify(response)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@main.route('/get_thumbnail', methods=['POST'])
def get_thumbnail():
    data = request.json
    song_name = data.get("song_name")
    thumbnail, video_url = get_youtube_thumbnail(song_name)
    return jsonify({"thumbnail": thumbnail, "video_url": video_url})
