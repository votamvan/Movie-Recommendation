from datetime import datetime
start_time = datetime.now()

import flask
from flask import request, jsonify
from flask_cors import CORS


from train import top10_movies, top_similar_movies, top10_recommend

app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True


API_RETURN = dict(
    LOGIN_SUCCESS = {
        "status": "success",
        "message": "login successfully",
        "data": {
            "userId": 44,
            "email": "obama@gmail.com",
            "displayName": "admin"
        }
    },
    LOGIN_FAIL = {
        "status": "error",
        "message": "wrong username or password",
        "data": None
    },
    RATING = {
        "status": "success",
        "data": 3.0
    },
    UPDATE_RATING_OK = {
        "status": "success"
    },
    UPDATE_RATING_FAIL = {
        "status": "error",
        "message": "userId, movieId or rating is missing"
    }
)

# dummy API
@app.route('/api/v0/rating/<userId>/<movieId>', methods=['POST'])
def rating(**kargs):
    print(kargs)
    return jsonify(API_RETURN["RATING"])

@app.route('/api/v0/updaterating', methods=['POST'])
def updaterating(**kargs):
    try:
        print(request.data)
        data = request.get_json()
        user_id = data["userId"]
        movie_id = data["movieId"]
        rating = data["rating"]
        return jsonify(API_RETURN["UPDATE_RATING_OK"])
    except:
        print("catch error")
    return jsonify(API_RETURN["UPDATE_RATING_FAIL"])

# real API
@app.route('/', methods=['GET'])
def home():
    return "<h1>Movie Recommendation API</h1>"

@app.route('/api/v0/login', methods=['POST'])
def login(**kargs):
    try:
        print(request.data)
        data = request.get_json()
        username = data["username"]
        password = data["password"]
        if username == "admin" and password == "cs582":
            return jsonify(API_RETURN["LOGIN_SUCCESS"])
        elif username == "thuy" and password == "cs582":
            out = {
                "status": "success",
                "message": "login successfully",
                "data": {
                    "userId": 45,
                    "email": "thuypham@miu.edu",
                    "displayName": "Thuy Pham"
                }
            }
            return jsonify(out)
    except:
        print("catch error")
    return jsonify(API_RETURN["LOGIN_FAIL"])

@app.route('/api/v0/toptrending/<nation>', methods=['POST'])
def toptrending(**kargs):
    print(kargs)
    top10 = top10_movies(cat="popular")
    api_return = {
        "status": "success",
        "data": top10
    }
    return jsonify(api_return)

@app.route('/api/v0/topsimilar/<movieId>', methods=['POST'])
def topsimilar(**kargs):
    print(kargs)
    movie_id = int(kargs["movieId"])
    data = top_similar_movies(movie_id)
    api_return = {
        "status": "success",
        "data": {
            "movie_details": data[0],
            "other_similar_movies": data[1:]
        }
    }
    return jsonify(api_return)

@app.route('/api/v0/recommend/<userId>', methods=['POST'])
def recommend(**kargs):
    print(kargs)
    user_id = int(kargs["userId"])
    top10 = top10_recommend(user_id)
    api_return = {
        "status": "success",
        "data": top10
    }
    return jsonify(api_return)

end_time = datetime.now()
print(f"Server started at {end_time.strftime('%Y-%m-%d %H:%M:%S')}, loading time = {end_time - start_time}")
app.run(host='0.0.0.0')
