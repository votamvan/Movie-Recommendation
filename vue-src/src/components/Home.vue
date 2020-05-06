<template>
    <div class="page">
      <div class="container-fluid">
          <h3 class="col-sm-12" v-if="topTrendingMovies.length>0">Top Trending Movies</h3>
          <div class="col-sm-2 mb-30" v-for="(api_data,index) in topTrendingMovies" :key="index">
              <router-link :to="{path: '/details/' + api_data.movieId}"><img class="movie-image img-responsive" :title="api_data.title" :src="api_data.url"></router-link>
          </div>
      </div>
      <div class="container-fluid">
          <h3 class="col-sm-12" v-if="userTopMovies.length>0">Top Recommendations For You</h3>
          <div class="col-sm-2 mb-30" v-for="(userTopMovie,index) in userTopMovies" :key="index">
              <router-link :to="{path: '/details/' + userTopMovie.movieId}"><img class="movie-image img-responsive" :title="userTopMovie.title" :src="userTopMovie.url"></router-link>
          </div>
      </div>
    </div>
</template>
<script type="text/javascript">
    export default {
      name: 'Home',
      components: {

      },
      data(){
        return {
          topTrendingMovies:[],
          userTopMovies:[]
        }
      },
      mounted(){
        /*home component is common for both guest user and after login user so displaying to trending movie for after login user only
        */
        if(this.helper.getUserInfo().userId>0){
              this.helper.request({
                    type: 'post',
                    auth:false,
                    withData:'json',
                    url: this.api.getTopTrendingMovieApi(),
                    dataType:'json',
                    success:(resp)=>{
                      if(resp.status == 'error'){
                        window.location.href = "/";
                      }else{
                        this.topTrendingMovies = resp.data;
                      }
                    }
              })
              this.helper.request({
                    type: 'post',
                    auth:false,
                    withData:'json',
                    url: this.api.getUserTopMoviesApi()+'/'+this.helper.getUserInfo().userId,
                    dataType:'json',
                    success:(resp)=>{
                      if(resp.status == 'error'){
                        window.location.href = "/";
                      }else{
                        this.userTopMovies = resp.data;
                      }
                    }
              })
        }
        

      }
    }
</script>