import pickle

# load all required files
with open("cosine_similarities_sbert.pkl", "rb") as f:
    cosine_similarities_sbert = pickle.load(f)

with open("indices.pkl", "rb") as f:
    indices = pickle.load(f)

with open("movie_title.pkl", "rb") as f:
    movie_title = pickle.load(f)


def content_recommender(title, num_movie=30):
    if title not in indices:
        raise KeyError("Title Not Found in database.")
    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarities_sbert[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_movie + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movie_title.iloc[movie_indices]
