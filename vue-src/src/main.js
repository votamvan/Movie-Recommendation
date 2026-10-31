import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import Api from './apis.js'
import Helper from './helper.js'
import 'bootstrap/dist/css/bootstrap.min.css'
import 'bootstrap-icons/font/bootstrap-icons.css'
import './assets/style.css'

const app = createApp(App)
app.config.globalProperties.api = new Api()
app.config.globalProperties.helper = new Helper()
app.use(router)
app.mount('#app-main')
