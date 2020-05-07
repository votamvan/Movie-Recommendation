<template>
  <div class="page mt-200">
    <!-- <h1>{{slug}}</h1>
    <p>{{content}}</p>-->

    <section id="services" class="container">
      <h2 class="display-4 text-center mt-1 mb-1">Movies on Cinema</h2>

      <div class="row text-center">
        <div class="col-md-4 mb-4">
          <div class="card h-100">
            <img
              style="width:50%"
              class="rounded float-left"
              src="https://upload.wikimedia.org/wikipedia/en/thumb/a/ab/Enemy_at_the_gates_ver2.jpg/220px-Enemy_at_the_gates_ver2.jpg"
              alt="movie 1"
            />
            <div class="card-body">
              <h4 class="card-title">Enemy At The Gate</h4>
              <p class="card-text">
                Vassili (Jude Law) is a young Russian sharpshooter who becomes
                a legend when a savvy polical officer (Joseph Fiennes) makes him the
                hero of his propaganda campaign. Their friendship is threatened when
                both men fall in love with a beautiful female soldier
              </p>
            </div>
            <div class="card-footer py-4">
              <a
                href="https://www.youtube.com/watch?v=4O-sMh_DO6I/"
                target="_blank"
                class="blinking"
              >See Trailer &raquo;</a>
            </div>
          </div>
        </div>

        <div class="col-md-4 mb-4">
          <div class="card h-100">
            <img
              style="width:50%"
              class="rounded float-left"
              src="https://images-na.ssl-images-amazon.com/images/I/51J7E5RVJDL._SY445_.jpg"
              alt="movie 1"
            />
            <div class="card-body">
              <h4 class="card-title">Behaind The Enemy Line</h4>
              <p class="card-text">
                US Navy helicopter pilot Lt Chris Burnett is part of the NATO force observing
                the peace treaty during the conflict in former Yugoslavia. Bored with routine duties,
                he flies off his designated route and spots
                Serb forces breaching the peace agreement, but they spot him too and down his craft.
              </p>
            </div>
            <div class="card-footer py-4">
              <a href="https://www.youtube.com/watch?v=A-yvybAaPks/" target="_blank" class="blinking" >See Trailer &raquo;</a>
               
               
              
            </div>
          </div>
        </div>
        <div class="col-md-4 mb-4">
          <div class="card h-100">
            <img
              style="width:50%"
              class="rounded float-left"
              src="https://images-na.ssl-images-amazon.com/images/I/51A348WXZ8L._AC_SY445_.jpg"
              alt="movie 1"
            />
            <div class="card-body">
              <h4 class="card-title">Message In A Bottle</h4>
              <p class="card-text">
                During her morning jog on the beach, journalist Theresa Osborne (Robin Wright Penn) discovers
                a bottle protruding from the sand. Inside it, she finds a heartbreaking, anonymous love letter.
              </p>
            </div>
            <div class="card-footer py-4">
              <a
                href="https://www.youtube.com/watch?v=Ur9xLzOJNIs/"
                target="_blank"
                class="blinking"
              >See Trailer &raquo;</a>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>
<script type="text/javascript">
export default {
  name: "Cms",
  components: {},
  data() {
    return {
      slug: "here",
      content: "",
      selectOne: "",
      count: 0,
      isActive: false,
      isActivetwo: false,
      isActivefour: false,
      isActivethree: false,
      isActivefive: false
    };
  },

  mounted() {
    this.slug = this.$route.params.sluging;
  },
  watch: {
    $route(to) {
      console.log(to);
      this.slug = this.$route.params.sluging;
    }
  }
};
</script>