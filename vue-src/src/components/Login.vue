<template>
  <div class="page mt-200">
    <div class="col-sm-4 offset-sm-4">
      <div class="card panel-theme">
        <div class="card-header">{{registering ? 'Create account' : 'Login'}}</div>
        <div class="card-body">
          <form @submit.prevent="doLogin">
            <p :class="alert_status" role="status">{{message}}</p>
            <div class="mb-3">
              <label for="username">Username</label>
              <input id="username" v-model.trim="username" class="form-control" required autocomplete="username">
            </div>
            <div class="mb-3">
              <label for="password">Password</label>
              <input id="password" v-model="password" type="password" class="form-control" required
                :minlength="registering ? 8 : 1" :autocomplete="registering ? 'new-password' : 'current-password'">
              <small v-if="registering">Use at least 8 characters.</small>
            </div>
            <button type="submit" class="btn btn-primary" :disabled="busy">{{registering ? 'Create account' : 'Login'}}</button>
            <button type="button" class="btn btn-link" :disabled="busy" @click="registering = !registering; message = ''">
              {{registering ? 'Back to login' : 'Create an account'}}
            </button>
          </form>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
export default {
  name: 'Login',
  data() {
    const message = this.helper.getLocalMessage();
    return { username: '', password: '', registering: false, busy: false,
      message: message.message || '', alert_status: message.alert_status };
  },
  methods: {
    doLogin() {
      this.busy = true;
      this.message = '';
      this.helper.request({
        url: this.registering ? this.api.getRegisterApi() : this.api.getLoginApi(),
        auth: false,
        data: { username: this.username, password: this.password },
        success: (response) => {
          this.alert_status = 'alert alert-success';
          this.message = response.message;
          if (this.registering) {
            this.registering = false;
            this.password = '';
          } else {
            this.helper.setUserInfo(response.data);
            window.location.href = '/';
          }
        },
        error: (response) => {
          this.alert_status = 'alert alert-danger';
          this.message = response.message;
        },
        complete: () => { this.busy = false; }
      });
    }
  }
};
</script>
