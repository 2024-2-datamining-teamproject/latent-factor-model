import pandas as pd
import numpy as np

predicted_ratings_df = pd.read_csv('predicted_ratings.csv', index_col=0)
ratings = pd.read_csv('ratings.csv') 

predicted_ratings = predicted_ratings_df.values
movie_ids = predicted_ratings_df.columns.astype(int)
user_ids = predicted_ratings_df.index.astype(int)

ratings_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
ratings_matrix = ratings_matrix.reindex(columns=movie_ids, fill_value=np.nan)
original_ratings_matrix = ratings_matrix.copy()

def recommend_by_prediction(user_id, top_n=10):
    user_idx = np.where(user_ids == user_id)[0][0]
    user_ratings = predicted_ratings[user_idx]
    
    rated_movies = original_ratings_matrix.loc[user_id][original_ratings_matrix.loc[user_id].notna()].index
    unrated_movies = [movie for movie in movie_ids if movie not in rated_movies]
    
    recommendations = {movie: user_ratings[np.where(movie_ids == movie)[0][0]] for movie in unrated_movies}
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    
    return [int(movie_id) for movie_id, _ in sorted_recommendations[:top_n]]

if __name__ == "__main__":
    user_id = 2
    print("Movie Recommendations Using Latent Factor Model:", recommend_by_prediction(user_id=user_id))
