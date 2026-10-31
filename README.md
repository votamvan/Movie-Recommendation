# Project 2: Movie-Recommendation

Historical research demo: http://104.154.84.211/ (may be unavailable).
For the maintained local setup, follow the instructions below.

Professor: Mohamed Abdelrazik

Team:
* Tam Van Vo - 610746   
* Quynh Pham - 610716   
* Samsher Bahadur Rana - 611060   
* Van Vong Tran - 610772   
* Yared Geberetsadik Beyene - 110466   

Project Colab URL: https://colab.research.google.com/drive/1bFKgCUKqSsoXzlMR2jA0N84506Wx1aB9   
Dataset: https://www.kaggle.com/rounakbanik/the-movies-dataset

## Run locally

Requirements: Python 3.10+ and Node.js 22 or 24 with npm.

Start the API in one terminal:

```sh
cd python-src
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app/main.py
```

Start the frontend in another terminal:

```sh
cd vue-src
npm ci
npm run serve
```

Open http://localhost:8080, choose **Login → Create an account**, then log in
with that account. The old shared demo credentials are no longer seeded.
The frontend development server proxies `/api` to `http://127.0.0.1:5000`.
Set `API_PROXY_TARGET` if the API runs elsewhere. For a production frontend
build, set `VUE_APP_API_URL` to the API origin or configure a same-origin
reverse proxy for `/api` (and a fallback to `index.html` for Vue routes).

Accounts, password hashes, seven-day sessions and ratings persist in
`python-src/instance/movies.sqlite3`, which is excluded from Git. Set
`MOVIE_DATABASE` to choose another database path. Rating `0` removes a rating;
positive ratings up to `5` are supported by the API. Logout revokes the session.
`PORT` changes the API port. `FRONTEND_ORIGIN` is a comma-separated CORS
allowlist for direct browser access; its default is `http://localhost:8080`.

The default recommender uses title, overview and genres from the included
`movies_metadata.csv`. Optional `credits.csv` and `keywords.csv` enrich results
when present; neither is required. New users receive popular movies, then
saved ratings influence content recommendations. Already-rated movies are
excluded. The original SVD experiment is preserved in `app/train_svd.py` with
its old dependencies in `requirements-svd.txt`; it is not loaded by the API.
It still requires its original datasets and a compatible legacy environment.
MovieLens and TMDB IDs require a `links.csv` mapping before integrating SVD
results. `/api/v0/predictrating` now returns JSON with HTTP 501 to make this
limitation explicit. Poster images are still loaded from the external image
URLs and require internet access.

## Docker

From the repository root:

```sh
docker compose up --build
```

Open http://localhost:8080. The API is exposed at http://localhost:5001
(container port 5000). A named Docker volume
keeps accounts and ratings across container restarts (`docker compose down`
preserves it; `docker compose down -v` removes it).

## Validation

GitHub Actions runs `.github/workflows/build.yml` for pull requests targeting
`master` and pushes to `master`. After the workflow is merged into `master`,
it can also be started manually from **Actions → Build and test → Run workflow**.
It runs API tests with Python 3.13 and frontend lint/build with Node.js 22, then
builds both Docker images if those checks pass. Jobs use standard Ubuntu
runners and require no additional secrets. Images are built for validation;
they are not published or deployed. Results and logs appear in the PR checks
and the Actions tab.

```sh
cd python-src
.venv/bin/python -m unittest discover -s tests -v
cd ../vue-src
npm run lint -- --no-fix
npm run build
```

API tests cover account validation, password hashing, wrong passwords,
session expiry/logout, authorization, rating persistence across app restarts,
rating removal, missing movies and recommendations without optional datasets.
To check the UI, create an account, try a wrong password, log in, rate a movie,
reload its detail page, restart the API, and confirm the rating remains.
Log out and verify that another account has its own ratings.
