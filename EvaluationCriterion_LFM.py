import pandas as pd

ratings = pd.read_csv('ratings.csv')
predicted_ratings = pd.read_csv('predicted_ratings_v5.csv', index_col=0)

base_user_id = 1

base_user_ratings = ratings[ratings['userId'] == base_user_id]

def evaluate_user(base_user_ratings, predicted_ratings, base_user_id):
    predicted_movie_ids = predicted_ratings.columns.astype(int)
    common_movies = base_user_ratings[base_user_ratings['movieId'].isin(predicted_movie_ids)]
    
    if common_movies.empty:
        return {
            'base_user_id': base_user_id,
            'total_common_movies': 0,
            'match_rate': 0.0
        }
    
    common_movies['predicted_rating'] = common_movies['movieId'].apply(
        lambda x: predicted_ratings.loc[base_user_id, str(x)] if str(x) in predicted_ratings.columns else None
    )
    
    common_movies['abs_diff'] = abs(common_movies['rating'] - common_movies['predicted_rating'])
    
    matches = common_movies[common_movies['abs_diff'] <= 1.0].shape[0]
    total_common_movies = common_movies.shape[0]
    
    match_rate = (matches / total_common_movies) * 100 if total_common_movies > 0 else 0.0
    
    return {
        'base_user_id': base_user_id,
        'total_common_movies': total_common_movies,
        'match_rate': match_rate
    }

if __name__ == "__main__":
    evaluation_result = evaluate_user(base_user_ratings, predicted_ratings, base_user_id)

    print("Evaluation Criterion of Latent Factor Model")
    print(f"Base User ID: {evaluation_result['base_user_id']}")
    print(f"Total Movies to Evaluate: {evaluation_result['total_common_movies']}")
    print(f"Prediction Accuracy Ratio: {evaluation_result['match_rate']:.2f}%")
