# Repository Guidelines

## Project Structure & Module Organization

- `vue-src/` contains the Vue 2 frontend. Page components live in `src/components/`; routing, API URLs, and request helpers are in `src/router.js`, `src/apis.js`, and `src/helper.js`. Static assets live in `src/assets/` and `public/`.
- `python-src/app/main.py` implements the Flask API; `app/train.py` handles preprocessing, content similarity, and Surprise SVD recommendations. The separate `python-src/main.py` is an older stub API.
- `python-src/data/` contains CSV datasets; `movie-posters/` contains poster images.
- `ML_Project2_final.ipynb` and its PDF document the research. API integration tests live in `python-src/tests/`.

## Build, Test, and Development Commands

Run commands from the indicated directory:

- `cd vue-src && npm install`: install frontend dependencies.
- `npm run serve`: start the frontend development server.
- `npm run build`: generate the production frontend bundle.
- `npm run lint`: run ESLint with Vue essential and recommended JavaScript rules; it may fix files.
- `cd python-src && python -m pip install -r requirements.txt`: install pinned backend dependencies in an isolated environment. The Dockerfile targets Python 3.13; local runtime supports Python 3.10+.
- `python app/main.py`: start Flask on port 5000. The default runtime loads metadata lazily without SVD training.
- In either source directory, `sh build.sh` builds its Docker image and `sh run.sh` starts it with mounted source. Frontend Docker exposes port 8081; backend exposes 5000.

Backend startup uses the included `movies_metadata.csv`. `credits.csv` and `keywords.csv` are optional enrichments. The original SVD experiment is retained in `app/train_svd.py` and `requirements-svd.txt`.

## Coding Style & Naming Conventions

Use four spaces and `snake_case` for Python functions and variables. Use PascalCase Vue component filenames and camelCase JavaScript methods. Existing frontend indentation varies; preserve surrounding style and avoid unrelated formatting. Run frontend lint before submitting changes.

## Testing Guidelines

Backend tests use the standard library unittest runner: `python -m unittest discover -s tests -v` from `python-src`. No coverage threshold is configured. For frontend changes, run lint and build, then manually check affected screens. For backend changes, exercise relevant `/api/v0/` endpoints and verify response fields and error handling. Record validation commands and results. SVD cross-validation reports RMSE/MAE but does not replace API testing.

## Commit & Pull Request Guidelines

History uses short messages such as `Update README.md` and `add login free`, with no enforced prefix convention. Write concise, imperative messages describing the change. PRs should explain behavior changes, validation, dataset or configuration requirements, and linked issues when applicable. Include screenshots for UI changes.

## Configuration Notes

The frontend development proxy targets `http://127.0.0.1:5000`; override it with `API_PROXY_TARGET`. Set `VUE_APP_API_URL` for builds that use a separate API origin. Accounts, sessions and ratings persist in ignored `python-src/instance/movies.sqlite3`. Do not commit real credentials or runtime databases.
