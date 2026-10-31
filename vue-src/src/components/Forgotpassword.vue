<template>
	<div class="page mt-200">
    <div class="col-sm-4 offset-sm-4">
      <div class="card panel-theme">
        <div class="card-header">Login</div>
        <div class="card-body">
          <div class="col-sm-12">
              <form class="form-theme" @submit="doLogin">
                <div class="row">
                  <div class="mb-3">
                    <label class="control-label">Username:</label>
                    <input type="text" v-model="username" name="username" class="form-control" placeholder="Username">
                  </div>
                  <div class="mb-3">
                    <label class="control-label">Password:</label>
                    <input type="password" v-model="password" name="password" class="form-control" placeholder="Password">
                  </div>
                  <div class="mb-3">
                    <router-link to="/forgot-password">Forgot your password?</router-link>
                  </div>
                  <div class="mb-3">
                    <button type="submit" class="btn btn-primary">Login</button>
                  </div>
                </div>
              </form>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script type="text/javascript">
	export default {
      name: 'Login',
      components: {
      },
      data(){
        return {
          errors:[],
          username:null,
          password:null
        }
      },
      methods:{
        doLogin:function(e){
          e.preventDefault();

          this.helper.request({
                type: 'post',
                withData:'json',
                url: this.api.getLoginApi(),
                dataType:'json',
                auth:false,
                data: {
                  username:this.username,
                  password:this.password
                },
                complete:()=>{
                  this.helper.showMessage('danger','Invalid Username or Password');
                },
                success:(resp)=>{
                  console.log(resp)
                }
          })
        }
      },
      mounted(){

      }
    }
</script>