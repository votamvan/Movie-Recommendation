import Vue from 'vue'
import App from './App.vue'
import router from './router'

Vue.config.productionTip = false

var Api = require('./apis.js');
let api = new Api();
Vue.prototype.api = api;

var Helper = require('./helper.js');
let helper = new Helper();
Vue.prototype.helper = helper;

new Vue({
	el: '#app-main',
	router:router,
  render: h => h(App),
})