from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import os
import requests

app = Flask(__name__)
CORS(app)

# ✅ Hugging Face raw links to your .pkl files (replace with yours)
movies_url = "https://huggingface.co/AhmadAlix/Movie_recommender/resolve/main/movies.pkl"
similarity_url = "https://huggingface.co/AhmadAlix/Movie_recommender/resolve/main/similarity.pkl"

# ✅ Download files if not already present
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

download_file(movies_url, "movies.pkl")
download_file(similarity_url, "similarity.pkl")

# ✅ Load the files
movies = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

# ✅ Recommendation API
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_name = data['movie_name']

    try:
        if movie_name not in movies['title'].values:
            return jsonify({"error": "Movie not found!"}), 404

        movie_index = movies[movies['title'] == movie_name].index[0]
        distances = similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = [movies.iloc[i[0]].title for i in movie_list]

        return jsonify({"recommended_movies": recommended_movies})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
