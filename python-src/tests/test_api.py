from contextlib import closing
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from main import create_app


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Path(self.directory.name) / 'test.sqlite3'
        self.client = create_app(self.database).test_client()
        self.account = {'username': 'alice', 'password': 'test-password'}
        self.assertEqual(self.client.post('/api/v0/register', json=self.account).status_code, 201)
        login = self.client.post('/api/v0/login', json=self.account).get_json()['data']
        self.user_id = login['userId']
        self.headers = {'Authorization': 'Bearer ' + login['userToken']}

    def tearDown(self):
        self.directory.cleanup()

    def post(self, path, data=None):
        return self.client.post('/api/v0/' + path, json=data, headers=self.headers)

    def test_login_validation_and_password_storage(self):
        self.assertEqual(self.client.post('/api/v0/login', json={**self.account, 'password': 'wrong'}).status_code, 401)
        self.assertEqual(self.client.post('/api/v0/register', json=self.account).status_code, 409)
        for body in [[], {}, {'username': 12, 'password': 'abc'}]:
            self.assertEqual(self.client.post('/api/v0/login', json=body).status_code, 400)
        with closing(sqlite3.connect(self.database)) as database, database:
            stored = database.execute('SELECT password_hash FROM users').fetchone()[0]
        self.assertNotEqual(stored, self.account['password'])

    def test_rating_survives_restart_and_is_isolated(self):
        url = f'rating/{self.user_id}/862'
        self.assertEqual(self.post(url).get_json()['data'], 0)
        data = {'userId': self.user_id, 'movieId': 862, 'rating': 4}
        self.assertEqual(self.post('updaterating', data).status_code, 200)
        self.client = create_app(self.database).test_client()
        self.assertEqual(self.post(url).get_json()['data'], 4)
        self.assertEqual(self.post('updaterating', {**data, 'rating': 5}).status_code, 200)
        self.assertEqual(self.post(url).get_json()['data'], 5)
        self.client.post('/api/v0/register', json={'username': 'bob', 'password': 'other-password'})
        bob = self.client.post('/api/v0/login', json={'username': 'bob', 'password': 'other-password'}).get_json()['data']
        bob_headers = {'Authorization': 'Bearer ' + bob['userToken']}
        self.assertEqual(self.client.post('/api/v0/' + url, headers=bob_headers).status_code, 403)
        self.assertEqual(self.client.post('/api/v0/updaterating', json=data, headers=bob_headers).status_code, 403)
        self.assertEqual(self.client.post(f"/api/v0/rating/{bob['userId']}/862", headers=bob_headers).get_json()['data'], 0)
        self.post('updaterating', {**data, 'rating': 0})
        self.assertEqual(self.post(url).get_json()['data'], 0)

    def test_auth_and_rating_validation(self):
        data = {'userId': self.user_id, 'movieId': 862, 'rating': 4}
        self.assertEqual(self.client.post('/api/v0/updaterating', json=data).status_code, 401)
        for value in [-1, 6, True, 'five', None, float('nan')]:
            self.assertEqual(self.post('updaterating', {**data, 'rating': value}).status_code, 400)
        for value in [True, 862.5, None, 'invalid']:
            self.assertEqual(self.post('updaterating', {**data, 'movieId': value}).status_code, 400)
        self.assertEqual(self.post('updaterating', {**data, 'movieId': 999999999}).status_code, 404)
        self.post('logout')
        self.assertEqual(self.post('updaterating', data).status_code, 401)

    def test_catalog_and_recommendations_without_optional_datasets(self):
        movies = self.post('toptrending/us').get_json()['data']
        self.assertEqual(len(movies), 10)
        detail = self.post('topsimilar/862').get_json()['data']
        self.assertEqual(detail['movie_details']['movieId'], 862)
        self.assertEqual(len(detail['other_similar_movies']), 10)
        by_name = self.post('topsimilarbyname', {'movie_name': 'Jumanji'})
        self.assertEqual(by_name.get_json()['data'][0]['movieId'], 8844)
        self.assertEqual(self.post('topsimilarbyname', {'movie_name': 'no-such-movie'}).status_code, 404)
        self.assertEqual(self.post('topsimilar/999999999').status_code, 404)
        self.assertEqual(self.post('topsimilar/invalid').status_code, 404)
        before = self.post(f'recommend/{self.user_id}').get_json()['data']
        self.post('updaterating', {'userId': self.user_id, 'movieId': 862, 'rating': 5})
        after = self.post(f'recommend/{self.user_id}').get_json()['data']
        self.assertEqual(len(after), 10)
        self.assertNotIn(862, [movie['movieId'] for movie in after])
        self.assertNotEqual(before, after)

    def test_request_limits_and_security_headers(self):
        response = self.client.post('/api/v0/login', json={**self.account, 'password': 'x' * 129})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
        response = self.client.post('/api/v0/login', json={'username': 'x' * 20000})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()['status'], 'error')
        for _ in range(10):
            response = self.client.post('/api/v0/login', json={**self.account, 'password': 'wrong'})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.get_json()['status'], 'error')

    def test_session_tokens_are_hashed_and_old_sessions_migrate(self):
        token = self.headers['Authorization'].removeprefix('Bearer ')
        with closing(sqlite3.connect(self.database)) as database, database:
            stored = database.execute('SELECT token FROM sessions').fetchone()[0]
            self.assertNotEqual(stored, token)
            database.execute('UPDATE sessions SET token = ?', (token,))
        self.client = create_app(self.database).test_client()
        self.assertEqual(self.post(f'recommend/{self.user_id}').status_code, 200)
        with closing(sqlite3.connect(self.database)) as database, database:
            stored = database.execute('SELECT token FROM sessions').fetchone()[0]
        self.assertNotEqual(stored, token)
        self.post('logout')
        self.assertEqual(self.post(f'recommend/{self.user_id}').status_code, 401)

    def test_expired_session(self):
        with closing(sqlite3.connect(self.database)) as database, database:
            database.execute('UPDATE sessions SET expires = 0')
        self.assertEqual(self.post(f'recommend/{self.user_id}').status_code, 401)


if __name__ == '__main__':
    unittest.main()
