<template>
  <div class="page mt-100">
    <div class="page">
      <div class="container-fluid">
        <div class="row">
          <div class="col-sm-10">
            <h3 class="col-sm-12 text-center">{{movie_details.title}}</h3>
            <div class="col-sm-8 mb-30">
              <img
                class="img-responsive"
                :alt="movie_details.title"
                :title="movie_details.title"
                :src="movie_details.url"
              />
            </div>
            <div class="col-sm-4 mb-30" style="font-size: 20px">
              <div class="row">
                <div class="col-sm-12">
                  <img
                    style="width:130px;float:left;margin-right:10px"
                    src="https://png.pngtree.com/png-vector/20190223/ourlarge/pngtree-vector-play-icon-png-image_695339.jpg"
                    alt="play btn"
                  />
                  <div>
                    <div class="form-group"><span class="text-bold">Score:</span> {{movie_details.score}}</div>
                    <div class="form-group"><span class="text-bold">Vote Count:</span> {{movie_details.vote_count}}</div>
                  </div>
                </div>
              </div>
              <div class="row">
                <div class="col-sm-12">
                  <div class="form-group mt-15">
                    <div class="pull-left form-group">Rate this movie</div>
                    <div class="pull-left" style="margin-left:5px">
                      <span role="status">{{ratingMessage}}</span>
                      <span
                        class="fa fa-star"
                        v-on:click="one"
                        v-bind:class="{ checked: count>=1 }"
                      ></span>
                      <span
                        class="fa fa-star"
                        v-on:click="two"
                        v-bind:class="{ checked: count>=2 }"
                      ></span>
                      <span
                        class="fa fa-star"
                        v-on:click="three"
                        v-bind:class="{ checked: count>=3 }"
                      ></span>
                      <span
                        class="fa fa-star"
                        v-on:click="four"
                        v-bind:class="{ checked: count>=4 }"
                      ></span>
                      <span
                        class="fa fa-star"
                        v-on:click="five"
                        v-bind:class="{ checked: count>=5 }"
                      ></span>
                    </div>
                  </div>
                </div>
              </div>
              <div class="row">
                <div class="col-sm-12">
                  <div class="form-group"><span class="text-bold">Director:</span> {{movie_details.director}}</div>
                  <div class="form-group"><span class="text-bold">Genres:</span> {{movie_details.genres.join(", ")}}</div>
                  <div class="form-group"><span class="text-bold">Tags:</span> {{movie_details.keywords.join(", ")}}</div>
                  <div class="form-group"><span class="text-bold">OverView:</span> {{movie_details.overview}}</div>
                </div>
              </div>
            </div>
          </div>
          <div class="row-sm-3">
            <h3 class="col-sm-12">Similar Movies</h3>
              <div class="col-sm-1" v-for="(api_data,index) in api_datas" :key="index">
                <div class="row-sm-3">
                <router-link :to="{path: '/details/' + api_data.movieId}">
                  <img
                    :alt="api_data.title"
                    class="movie-image img-responsive"
                    :title="api_data.title"
                    :src="api_data.url"
                  />
                </router-link>
              </div>
            
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script type="text/javascript">
export default {
  name: "ItemDetails",
  components: {},
  data() {
    return {
      movie_details: { genres: [], keywords: [] },
      ratingBusy: false,
      ratingReady: false,
      ratingMessage: '',
      api_datas: [],
      movieId: 0,
      count: 0,
      userId: this.helper.getUserInfo().userId
    };
  },
  watch: {
    $route(to) {
      console.log(to);
      this.callApi();
    }
  },
  methods: {
    async callApi() {
      const movieId = this.$route.params.id;
      this.movieId = movieId;
      this.count = 0;
      this.ratingReady = false;
      this.ratingMessage = '';
      await this.helper.request({
        url: this.api.getTopSimilarMovieApi() + '/' + movieId,
        success: (response) => {
          if (this.$route.params.id !== movieId) return;
          this.movie_details = response.data.movie_details;
          this.api_datas = response.data.other_similar_movies;
        },
        error: (response) => {
          if (this.$route.params.id === movieId) this.ratingMessage = response.message;
        }
      });
      if (this.$route.params.id !== movieId) return;
      await this.helper.request({
        url: this.api.getRatingApi() + '/' + this.userId + '/' + movieId,
        success: (response) => {
          if (this.$route.params.id !== movieId) return;
          this.count = response.data;
          this.ratingReady = true;
        },
        error: (response) => {
          if (this.$route.params.id === movieId) this.ratingMessage = response.message;
        }
      });
    },
    async setRating(value) {
      if (this.ratingBusy || !this.ratingReady) return;
      const movieId = this.$route.params.id;
      this.ratingBusy = true;
      this.ratingMessage = 'Saving...';
      const next = this.count === value ? 0 : value;
      await this.helper.request({
        url: this.api.getUpdateRatingApi(),
        data: { userId: this.userId, movieId, rating: next },
        success: (response) => {
          if (this.$route.params.id !== movieId) return;
          this.count = response.data;
          this.ratingMessage = 'Saved';
        },
        error: (response) => {
          if (this.$route.params.id === movieId) this.ratingMessage = response.message;
        }
      });
      this.ratingBusy = false;
    },
    one() { this.setRating(1); },
    two() { this.setRating(2); },
    three() { this.setRating(3); },
    four() { this.setRating(4); },
    five() { this.setRating(5); }
  },
  mounted() {
    if (!this.userId || !this.helper.getUserInfo().userToken) {
      window.location.href = "/login";
      return;
    }
    this.callApi();
  }
};
</script>