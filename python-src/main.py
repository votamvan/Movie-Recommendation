import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True


API_RETURN = dict(
    TOP_TREND = {
        "status": "success",
        "data": [1371, 2150, 2968, 265, 300, 370, 508, 593, 1721, 2959]
    },
    TOP_SIMILAR = {
        "status": "success",
        "data": [48783, 58559, 7153, 1298, 1387, 1917, 7153, 551, 720, 736]
    },
    RATING = {
        "status": "success",
        "data": 0.85
    }
)

@app.route('/', methods=['GET'])
def home():
    return "<h1>Movie Recommendation API</h1>"

@app.route('/api/v0/toptrending/<nation>', methods=['POST'])
def toptrending(**kargs):
    print(kargs)
    return jsonify(API_RETURN["TOP_TREND"])

@app.route('/api/v0/topsimilar/<movieId>', methods=['POST'])
def topsimilar(**kargs):
    print(kargs)
    return jsonify(API_RETURN["TOP_SIMILAR"])

@app.route('/api/v0/rating/<userId>/<movieId>', methods=['POST'])
def rating(**kargs):
    print(kargs)
    return jsonify(API_RETURN["RATING"])

#app.run()


