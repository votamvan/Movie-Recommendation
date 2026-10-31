<template>
  <nav class="navbar navbar-expand navbar-dark fixed-top navbar-theme">
    <div class="container flex-wrap">
      <router-link class="navbar-brand" to="/">Home</router-link>
      <ul class="navbar-nav ms-auto flex-row gap-3 align-items-center">
        <li class="nav-item"><router-link class="nav-link" to="/cms/privacy-policy">Now On Cinema</router-link></li>
        <li class="nav-item"><router-link class="nav-link" to="/about">About</router-link></li>
        <li v-if="!displayName" class="nav-item"><router-link class="nav-link" to="/login">Login</router-link></li>
        <li v-else class="nav-item dropdown">
          <button class="nav-link dropdown-toggle btn" :aria-expanded="dropdownOpen" @click="dropdownOpen = !dropdownOpen">
            {{displayName}}
          </button>
          <ul class="dropdown-menu dropdown-menu-end" :class="{ show: dropdownOpen }">
            <li><a href="#" class="dropdown-item" @click="logout">Logout</a></li>
          </ul>
        </li>
      </ul>
    </div>
  </nav>
</template>
<script>
export default {
  name: 'SiteHeader',
  data() {
    return { dropdownOpen: false, displayName: this.helper.getUserInfo().displayName };
  },
  methods: {
    logout(event) {
      event.preventDefault();
      this.helper.request({
        url: this.api.getApiUrl() + '/logout',
        complete: () => {
          this.helper.unsetUserInfo();
          this.helper.setLocalMessage('success', 'Logout successfully.');
          window.location.href = '/login';
        }
      });
    }
  }
};
</script>
