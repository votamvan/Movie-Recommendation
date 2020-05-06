<template>
	<div class="page mt-200">
    <div class="col-sm-4 col-sm-offset-4">
      <div class="panel panel-default panel-theme">
        <div class="panel-heading">Login</div>
        <div class="panel-body">
          <div class="col-sm-12">
              <form class="form-theme" @submit="doLogin">
                <div class="row">
                  <p :class="alert_status">{{message}}</p>
                </div>
                <div class="row">
                  <div class="form-group">
                    <label class="control-label">Username:</label>
                    <input type="text" v-model="username" name="username" class="form-control" placeholder="Username">
                    <div class="text-danger">{{errors.username}}</div>
                  </div>
                  <div class="form-group">
                    <label class="control-label">Password:</label>
                    <input type="password" v-model="password" name="password" class="form-control" placeholder="Password">
                    <div class="text-danger">{{errors.password}}</div>
                  </div>
                  <div class="form-group">
                    <router-link to="/forgot-password">Forgot your password?</router-link>
                  </div>
                  <div class="form-group">
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
        // initializing the datas
        var messageBox = this.helper.getLocalMessage();
        return {
          errors:{
            username:'',
            password:''
          },
          username:null,
          password:null,
          message:messageBox.message,
          alert_status:messageBox.alert_status
        }
      },
      methods:{
        doLogin:function(e){
          e.preventDefault();
          this.errors.username = '';
          this.errors.password = '';
          var errorCnt = 0;
          if(!this.username){
              this.errors.username = "Username field is required.";
              errorCnt++;
          }
          if(!this.password){
              this.errors.password = "Password field is required.";
              errorCnt++;
          }
          if(errorCnt == 0){
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
                      if(resp.status == 'error'){
                        this.message = resp.message;
                        this.alert_status = "alert alert-danger";
                      }else{
                        this.alert_status = "alert alert-success";
                        this.message = resp.message;
                        this.helper.setUserInfo(resp.data);
                        setTimeout(()=>{ 
                          window.location.href = "/";                                      
                        }, 2000);
                      }
                    }
              })
          }
          
        }
      },
      mounted(){
        if(this.helper.getUserInfo().displayName != ""){
          window.location.href = "/";
        }
      }
    }
</script>