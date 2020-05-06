module.exports = class Api {
	getHostUrl(){
		// return 'https://mighty-escarpment-32961.herokuapp.com';
		return 'http://ec2-18-222-249-229.us-east-2.compute.amazonaws.com:5000';
	}
	getApiVersion(){
		return 'v0';
	}
	getApiUrl(){
		return this.getHostUrl()+'/api'+'/'+this.getApiVersion();
	}
	
	getRegisterApi(){
		return this.getApiUrl()+'/register';
	}
	getLoginApi(){
		return this.getApiUrl()+'/login';
	}
	getUserInfoApi(){
		return this.getApiUrl()+'/users';//+{id} with get request
	}
	getTopTrendingMovieApi(){
		return this.getApiUrl()+'/toptrending/us';
	}
	getTopSimilarMovieApi(){
		return this.getApiUrl()+'/topsimilar';//+{movieId}
	}
	getUserTopMoviesApi(){
		return this.getApiUrl()+'/recommend';//+{userId}
	}
	getUpdateRatingApi(){
		return this.getApiUrl()+'/updaterating';
	}
	getRatingApi(){
		return this.getApiUrl()+'/rating';//+{userId}+{movieId}
	}
	
}