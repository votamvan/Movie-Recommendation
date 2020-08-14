module.exports = class Api {
	getHostUrl(){
		return 'http://104.154.84.211:5000';
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