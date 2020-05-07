from collections import defaultdict
import pandas as pd 
import numpy as np
import os
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate


BASE_URL = "https://raw.githubusercontent.com/votamvan/Movie-Recommendation/master/movie-posters/"

# Load csv file
dir_path = os.path.dirname(os.path.realpath(__file__))
path = dir_path + "/../data/"
credit_df = pd.read_csv(path + "credits.csv")
movie_df = pd.read_csv(path + "movies_metadata.csv")
keyword_df = pd.read_csv(path + "keywords.csv")
rating_df = pd.read_csv(path + 'ratings_small.csv')
rating_df.userId = pd.to_numeric(rating_df.userId)
rating_df.movieId = pd.to_numeric(rating_df.movieId)

# pre-processing data
keyword_df['id'] = keyword_df['id'].astype('int')
movie_df.drop(movie_df[movie_df['imdb_id'] == '0'].index, inplace=True)
movie_df.dropna(subset=['imdb_id'], inplace=True)
movie_df['overview'] = movie_df['overview'].fillna('')
movie_df.id = pd.to_numeric(movie_df.id)
movie_df = movie_df.merge(credit_df, on='id')
movie_df = movie_df.merge(keyword_df, on='id')

# calculate weighted score
C = movie_df['vote_average'].mean()
m = movie_df['vote_count'].quantile(0.9)

def weighted_rating(x, m=m, C=C):
    """ Calculation based on the IMDB formula """
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

movie_df['score'] = movie_df.apply(weighted_rating, axis=1)


for key in ['cast', 'crew', 'keywords', 'genres']:
    movie_df[key] = movie_df[key].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return ''

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3: names = names[:3]
        return names
    return []

movie_df['director'] = movie_df['crew'].apply(get_director)
for col in ['cast', 'keywords', 'genres']:
    movie_df[col] = movie_df[col].apply(get_list)

# create new feauture SOUP
def create_soup(df):
    def _clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        return str.lower(x.replace(" ", ""))

    def _create_soup(x):
        soup = ""
        if len(x['keywords']) > 0:
          soup += ' '.join(x['keywords']) + ' '
        if len(x['cast']) > 0:
          soup += ' '.join(x['cast']) + ' '
        soup += x['director'] + ' '
        if len(x['genres']) > 0:
          soup += ' '.join(x['genres'])
        return soup

    temp_df = pd.DataFrame()
    for col in ['cast', 'keywords', 'director', 'genres']:
        temp_df[col] = movie_df[col].apply(_clean_data)
    return temp_df.apply(_create_soup, axis=1)

movie_df['soup'] = create_soup(movie_df)
count_matrix = CountVectorizer(stop_words='english').fit_transform(movie_df['soup'])
# tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(movie_df['soup'])

def _top_similar_movie(movie_id, df, sim_matrix):
    idx = df[df.id == movie_id].index[0]
    sim_scores = cosine_similarity(sim_matrix[idx], sim_matrix)[0]
    movie_indices = sorted(range(len(sim_scores)), key=lambda i: sim_scores[i], reverse=True)[:11]
    if idx not in movie_indices:
        movie_indices.append(idx)
    top10 = df.iloc[movie_indices]
    for index, row in top10.iterrows():
        print(sim_scores[index], row.title, row.imdb_id)
    return top10

def top_similar_movies(movie_id):
    df = _top_similar_movie(movie_id, movie_df, sim_matrix=count_matrix)
    return df2api(df)


def _top_rating(df):
    m = df['vote_count'].quantile(0.9)
    q_movies = df.copy().loc[df['vote_count'] >= m]
    q_movies = q_movies.sort_values('score', ascending=False)
    return q_movies.head(10)

def _top_popular(df):
    df.popularity = pd.to_numeric(df.popularity)
    pop = df.sort_values('popularity', ascending=False)
    return pop.head(10)

def top10_movies(cat="popular"):
    if cat == "popular":
        df = _top_popular(movie_df)
    elif cat == "rating":
        df = _top_rating(movie_df)
    return df2api(df)


def df2api(df):
    columns=['id', 'imdb_id', 'title', 'vote_count', 'vote_average', 'score', 'director', 'overview', 'genres', 'keywords']
    top10 = []
    for index, row in df.iterrows():
        rec_json = {}
        for col in columns:
            if col == 'id':
                rec_json.update({"movieId" : row[col]})
            elif col == 'score':
                rec_json.update({col: round(row[col], 1)})
            else:
                rec_json.update({col: row[col]})
        imdb_id = rec_json["imdb_id"]
        rec_json.update(url = BASE_URL + f"{imdb_id}.jpg")
        top10.append(rec_json)
    return top10

# SVD recommendation algorithm
reader = Reader()
data = Dataset.load_from_df(rating_df[['userId', 'movieId', 'rating']], reader)
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
trainset = data.build_full_trainset()
svd.fit(trainset)
testset = trainset.build_anti_testset()
predictions = svd.test(testset)

def get_top_n(predictions, n=15):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

top10_svd = get_top_n(predictions)

def _top10_recommend(user_id, df):
    movie_indices = []
    ids = df.id.unique()
    for movie_id, rating in top10_svd[user_id]:
        print(movie_id, rating)
        if movie_id in ids:
            movie_indices.append(df[df.id == movie_id].index[0])
    return df.iloc[movie_indices]

def top10_recommend(user_id):
    top10 = _top10_recommend(user_id, movie_df)
    return df2api(top10)


def top10_recommend_by_name(movie_name):
    df = movie_df[movie_df.title == movie_name].id
    if df.shape[0] == 0:
        return {
            "status": "error",
            "message": "movie name not found"
        }
    return top_similar_movies(df[0])


def predict_rating(user_id, movie_id):
    if user_id not in top10_svd:
        return {
            "status": "error",
            "message": "user id not found"
        }
    pred = svd.predict(user_id, movie_id)
    print(pred)
    return {
        "status": "success",
        "data": pred.est
    }


