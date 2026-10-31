"""Local movie API with persistent accounts, sessions and ratings."""
from contextlib import contextmanager
import math
import os
import secrets
import sqlite3
import time
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash
from train import catalog, top10_movies, top10_recommend, top_similar_movies
from train import top10_recommend_by_name


def create_app(database_path=None):
    app = Flask(__name__)
    CORS(app, origins=os.environ.get('FRONTEND_ORIGIN', 'http://localhost:8080').split(','))
    app.config['DATABASE'] = str(database_path or os.environ.get('MOVIE_DATABASE') or
                               Path(__file__).resolve().parent.parent / 'instance' / 'movies.sqlite3')
    Path(app.config['DATABASE']).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect():
        connection = sqlite3.connect(app.config['DATABASE'])
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    with connect() as database:
        database.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL, display_name TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS ratings (
                user_id INTEGER NOT NULL, movie_id INTEGER NOT NULL, rating REAL NOT NULL,
                PRIMARY KEY (user_id, movie_id));
        ''')

    def error(message, status=400):
        return jsonify(status='error', message=message), status

    def current_user():
        token = request.headers.get('Authorization', '').removeprefix('Bearer ')
        with connect() as database:
            return database.execute('''SELECT users.* FROM users JOIN sessions
                ON users.id = sessions.user_id WHERE token = ? AND expires > ?''',
                                    (token, time.time())).fetchone()

    def payload():
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else {}

    @app.get('/')
    def home():
        return jsonify(status='success', message='Movie Recommendation API')

    @app.post('/api/v0/register')
    def register():
        data = payload()
        username, password = data.get('username'), data.get('password')
        if not isinstance(username, str) or not 3 <= len(username.strip()) <= 80:
            return error('Username must contain 3 to 80 characters')
        if not isinstance(password, str) or not 8 <= len(password) <= 128:
            return error('Password must contain 8 to 128 characters')
        username = username.strip()
        try:
            with connect() as database:
                database.execute('INSERT INTO users (username, password_hash, display_name) VALUES (?, ?, ?)',
                                 (username, generate_password_hash(password), username))
        except sqlite3.IntegrityError:
            return error('Username already exists', 409)
        return jsonify(status='success', message='Account created. Please log in.'), 201

    @app.post('/api/v0/login')
    def login():
        data = payload()
        username, password = data.get('username'), data.get('password')
        if not isinstance(username, str) or not isinstance(password, str):
            return error('Username and password are required')
        with connect() as database:
            user = database.execute('SELECT * FROM users WHERE username = ?', (username.strip(),)).fetchone()
            if user is None or not check_password_hash(user['password_hash'], password):
                return error('Wrong username or password', 401)
            token = secrets.token_urlsafe(32)
            database.execute('DELETE FROM sessions WHERE expires <= ?', (time.time(),))
            database.execute('INSERT INTO sessions VALUES (?, ?, ?)',
                             (token, user['id'], time.time() + 7 * 86400))
        return jsonify(status='success', message='Login successfully', data=dict(
            userId=user['id'], displayName=user['display_name'], email='', userToken=token))

    @app.post('/api/v0/logout')
    def logout():
        token = request.headers.get('Authorization', '').removeprefix('Bearer ')
        with connect() as database:
            database.execute('DELETE FROM sessions WHERE token = ?', (token,))
        return jsonify(status='success')

    @app.post('/api/v0/rating/<int:user_id>/<int:movie_id>')
    def rating(user_id, movie_id):
        user = current_user()
        if user is None:
            return error('Please log in', 401)
        if user['id'] != user_id:
            return error('Access denied', 403)
        if movie_id not in catalog():
            return error('Movie not found', 404)
        with connect() as database:
            row = database.execute('SELECT rating FROM ratings WHERE user_id = ? AND movie_id = ?',
                                   (user_id, movie_id)).fetchone()
        return jsonify(status='success', data=row['rating'] if row else 0)

    @app.post('/api/v0/updaterating')
    def update_rating():
        user = current_user()
        if user is None:
            return error('Please log in', 401)
        data = payload()
        try:
            identifiers = [data['userId'], data['movieId']]
            if any(isinstance(value, bool) or not str(value).isdigit() for value in identifiers):
                raise ValueError()
            user_id, movie_id = map(int, identifiers)
            value = data['rating']
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError()
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 5:
                raise ValueError()
        except (KeyError, TypeError, ValueError, OverflowError):
            return error('Valid userId, movieId and rating (0 to 5) are required')
        if user['id'] != user_id:
            return error('Access denied', 403)
        if movie_id not in catalog():
            return error('Movie not found', 404)
        with connect() as database:
            if value == 0:
                database.execute('DELETE FROM ratings WHERE user_id = ? AND movie_id = ?', (user_id, movie_id))
            else:
                database.execute('INSERT INTO ratings VALUES (?, ?, ?) ON CONFLICT(user_id, movie_id) '
                                 'DO UPDATE SET rating = excluded.rating', (user_id, movie_id, value))
        return jsonify(status='success', data=value)

    @app.post('/api/v0/toptrending/<nation>')
    def trending(nation):
        return jsonify(status='success', data=top10_movies())

    @app.post('/api/v0/topsimilar/<int:movie_id>')
    def similar(movie_id):
        if movie_id not in catalog():
            return error('Movie not found', 404)
        movies = top_similar_movies(movie_id)
        return jsonify(status='success', data=dict(movie_details=movies[0], other_similar_movies=movies[1:]))

    @app.post('/api/v0/recommend/<int:user_id>')
    def recommend(user_id):
        user = current_user()
        if user is None:
            return error('Please log in', 401)
        if user['id'] != user_id:
            return error('Access denied', 403)
        with connect() as database:
            rows = database.execute('SELECT movie_id, rating FROM ratings WHERE user_id = ?', (user_id,))
            ratings = {row['movie_id']: row['rating'] for row in rows}
        return jsonify(status='success', data=top10_recommend(ratings))

    @app.post('/api/v0/topsimilarbyname')
    def similar_by_name():
        name = payload().get('movie_name')
        if not isinstance(name, str) or not name.strip():
            return error('Movie name is required')
        try:
            return jsonify(status='success', data=top10_recommend_by_name(name))
        except KeyError:
            return error('Movie not found', 404)

    @app.post('/api/v0/predictrating')
    def predict_rating():
        return error('SVD prediction is available only in the optional research pipeline', 501)

    @app.errorhandler(404)
    def not_found(_error):
        return error('Resource not found', 404)

    return app


if __name__ == '__main__':
    create_app().run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
