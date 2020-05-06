<template>
    <!-- Footer -->


<footer    class="page-footer font-small blue pt-4" >


  <!-- Footer Links -->
  <div class="container-fluid text-center text-md-left">

    <!-- Grid row -->
    <div class="row">


      <hr class="clearfix w-100 d-md-none pb-3">

      <!-- Grid column -->
      <div class="col-md-5 mb-md-0 mb-3 col-md-offset-7">

        <!-- Links -->
        <h5 class="text-uppercase">Other Links</h5>

        <ul class="list-unstyled">
          <li>
            <router-link to="/cms/privacy-policy">Privacy and Policy</router-link>
          </li>
          <li>
            <router-link to="/cms/about">About</router-link>
          </li>
          <li>
            <router-link to="/contact-us">Contact Us</router-link>
          </li>
        </ul>

      </div>
      <!-- Grid column -->

    </div>
    <!-- Grid row -->

  </div>
  <!-- Footer Links -->

  <!-- Copyright -->
  <div class="footer-copyright text-center py-3">© 2020 Copyright:
    <router-link to="/">home.com</router-link>
  </div>
  <!-- Copyright -->

</footer>

<!-- Footer -->
</template>
<script type="text/javascript">
    export default {
  name: 'App',
  components: {
  },
  data(){
    return {
      api_data:[]
    }
  },
  mounted(){
    this.api_data = [1,2,3];
  }
}
</script>