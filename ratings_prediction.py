import pandas as pd
import numpy as np

latent_dim = 50
learning_rate = 0.001
lambda_reg = 0.01
epochs = 100

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

all_movie_ids = movies['movieId'].unique()

user_ids = ratings['userId'].unique()
user_id_map = {id_: idx for idx, id_ in enumerate(user_ids)}
movie_id_map = {id_: idx for idx, id_ in enumerate(all_movie_ids)}

ratings['user_idx'] = ratings['userId'].map(user_id_map)
ratings['movie_idx'] = ratings['movieId'].map(movie_id_map)

num_users = len(user_ids)
num_movies = len(all_movie_ids)

P = np.random.normal(scale=0.1, size=(num_users, latent_dim))
Q = np.random.normal(scale=0.1, size=(num_movies, latent_dim))

RATING_MIN = ratings['rating'].min()
RATING_MAX = ratings['rating'].max()
ratings['rating_normalized'] = (ratings['rating'] - RATING_MIN) / (RATING_MAX - RATING_MIN)

for epoch in range(epochs):
    total_loss = 0
    for _, row in ratings.iterrows():
        user_idx = int(row['user_idx'])
        movie_idx = int(row['movie_idx'])
        rating = row['rating_normalized']

        pred_rating = np.dot(P[user_idx], Q[movie_idx])
        error = rating - pred_rating

        P[user_idx] += learning_rate * (error * Q[movie_idx] - lambda_reg * P[user_idx])
        Q[movie_idx] += learning_rate * (error * P[user_idx] - lambda_reg * Q[movie_idx])
        
        total_loss += error**2 + lambda_reg * (np.linalg.norm(P[user_idx])**2 + np.linalg.norm(Q[movie_idx])**2)
        
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    if epoch == 0 or (epoch + 1) % 10 == 0:
        print(f"Sample Error: {error:.4f}, Predicted Rating: {pred_rating:.4f}, Actual Rating: {rating:.4f}")


if __name__ == "__main__":
    print("Calculating all predicted ratings...")
    predicted_ratings = np.dot(P, Q.T)
    predicted_ratings = predicted_ratings * (RATING_MAX - RATING_MIN) + RATING_MIN
    predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_ids, columns=all_movie_ids)
    predicted_ratings_df = predicted_ratings_df.clip(lower=0, upper=5)
    predicted_ratings_df.to_csv('predicted_ratings_v5.csv')
    print("Predicted ratings saved to 'predicted_ratings.csv'.")

