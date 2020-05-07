<template>
  <nav class="navbar navbar-default navbar-fixed-top navbar-theme">
    <div class="container">
      <div class="navbar-header">
        <router-link class="navbar-brand" to="/">
          <h3 style="margin-left:700px;margin-top:1px;padding:0px">Home</h3 >
        </router-link>
      </div>
      <ul class="nav navbar-nav navbar-right">
        <li>
          <router-link to="/cms/privacy-policy">Now On Cinema</router-link>
        </li>
        <li>
          <router-link to="/about">About</router-link>
        </li>
        <li v-if="displayName == '' ">
          <router-link to="/login">Login</router-link>
        </li>
        <li v-if="displayName != '' " class="dropdown">
          <a href="#" class="dropdown-toggle" data-toggle="dropdown">
            {{displayName}}
            <span class="caret"></span>
          </a>
          <ul class="dropdown-menu">
            <!-- <li>
              <a href="#">Profile</a>
            </li> -->
            <li>
              <a href="#" v-on:click="logout">Logout</a>
            </li>
          </ul>
        </li>
      </ul>
    </div>
  </nav>
</template>
<script>
export default {
  name: "Header",
  props: {},
  data() {
    return {
      displayName: this.helper.getUserInfo().displayName
    };
  },
  methods: {
    logout: function(e) {
      this.helper.unsetUserInfo();
      this.helper.setLocalMessage("success", "Logout successfully.");
      window.location.href = "/login";
      e.preventDefalt();
    }
  },
  mounted() {}
};
</script>
