"""Metadata recommendations. Original optional SVD research is in train_svd.py.
MovieLens rating IDs cannot be joined to TMDB IDs without links.csv.
"""
import ast
import csv
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / 'data'
POSTER_URL = 'https://raw.githubusercontent.com/votamvan/Movie-Recommendation/master/movie-posters/'


def parse_list(value):
    try:
        result = ast.literal_eval(value or '[]')
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


@lru_cache(maxsize=1)
def catalog():
    movies = {}
    with (DATA_PATH / 'movies_metadata.csv').open(encoding='utf-8', newline='') as source:
        for row in csv.DictReader(source):
            try:
                movie_id = int(row['id'])
                votes = int(float(row['vote_count'] or 0))
                average = float(row['vote_average'] or 0)
                popularity = float(row['popularity'] or 0)
            except (ValueError, TypeError):
                continue
            if not row['title'] or not row['imdb_id']:
                continue
            movies[movie_id] = {
                'movieId': movie_id, 'imdb_id': row['imdb_id'], 'title': row['title'],
                'overview': row['overview'] or '',
                'genres': [item['name'] for item in parse_list(row['genres'])],
                'keywords': [], 'director': '', 'vote_count': votes, 'vote_average': average,
                'score': round((votes * average + 100 * 5.5) / (votes + 100), 1),
                'url': POSTER_URL + row['imdb_id'] + '.jpg', '_popularity': popularity,
            }
    for filename, field in [('keywords.csv', 'keywords'), ('credits.csv', 'crew')]:
        optional = DATA_PATH / filename
        if not optional.exists():
            continue
        with optional.open(encoding='utf-8', newline='') as source:
            for row in csv.DictReader(source):
                try:
                    movie = movies[int(row['id'])]
                except (ValueError, KeyError):
                    continue
                items = parse_list(row.get(field))
                if field == 'keywords':
                    movie['keywords'] = [item['name'] for item in items]
                else:
                    movie['director'] = next((item['name'] for item in items
                                              if item.get('job') == 'Director'), '')
    for movie in movies.values():
        text = ' '.join([movie['title'], movie['overview'], movie['director']]
                        + movie['genres'] * 3 + movie['keywords'])
        counts = Counter(re.findall(r'\w{3,}', text.lower()))
        norm = math.sqrt(sum(value * value for value in counts.values())) or 1
        movie['_vector'] = {key: value / norm for key, value in counts.items()}
    return movies


def public(movie):
    return {key: value for key, value in movie.items() if not key.startswith('_')}


def similarity(first, second):
    return sum(value * second.get(key, 0) for key, value in first.items())


def top10_movies(cat='popular'):
    key = '_popularity' if cat == 'popular' else 'score'
    return [public(movie) for movie in sorted(catalog().values(),
            key=lambda movie: movie[key], reverse=True)[:10]]


def top_similar_movies(movie_id):
    selected = catalog()[movie_id]
    others = sorted((movie for movie in catalog().values() if movie['movieId'] != movie_id),
                    key=lambda movie: similarity(selected['_vector'], movie['_vector']),
                    reverse=True)[:10]
    return [public(selected)] + [public(movie) for movie in others]


def top10_recommend(ratings):
    movies = catalog()
    profile = Counter()
    for movie_id, rating in ratings.items():
        if movie_id in movies:
            for word, value in movies[movie_id]['_vector'].items():
                profile[word] += (rating - 3) * value
    candidates = [movie for movie in movies.values() if movie['movieId'] not in ratings]
    candidates.sort(key=lambda movie: (similarity(profile, movie['_vector']),
                                      movie['_popularity']), reverse=True)
    return [public(movie) for movie in candidates[:10]]


def top10_recommend_by_name(movie_name):
    movie = next((movie for movie in catalog().values()
                  if movie['title'].casefold() == movie_name.strip().casefold()), None)
    if movie is None:
        raise KeyError(movie_name)
    return top_similar_movies(movie['movieId'])
