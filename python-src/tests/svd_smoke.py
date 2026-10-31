"""Check that the optional scientific stack can train and predict together."""
import math


def main():
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from surprise import Dataset, Reader, SVD

    ratings = pd.DataFrame([
        (1, 10, 4.0), (1, 20, 3.0), (2, 10, 5.0), (2, 20, 2.0)
    ], columns=['userId', 'movieId', 'rating'])
    data = Dataset.load_from_df(ratings, Reader(rating_scale=(0, 5)))
    model = SVD(random_state=0, n_epochs=2).fit(data.build_full_trainset())
    estimate = model.predict(1, 20).est
    if not math.isfinite(estimate) or not 0 <= estimate <= 5:
        raise RuntimeError('Invalid SVD prediction')
    vectors = CountVectorizer().fit_transform(['space adventure', 'space story'])
    if not np.isclose(cosine_similarity(vectors[0], vectors[0])[0, 0], 1):
        raise RuntimeError('Invalid content similarity')
    print(f'SVD/scientific stack smoke check passed (prediction={estimate:.3f})')


if __name__ == '__main__':
    main()
