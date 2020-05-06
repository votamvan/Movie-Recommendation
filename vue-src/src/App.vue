<template>

  <div :class="fill_bg">
    <Header/>
    <section class="main-body"><router-view/></section>
    <Footer/>
  </div>
</template>

<script>

import Header from './components/Header.vue'
import Footer from './components/Footer.vue'

import './assets/style.css'
export default {
  name: 'App',
  components: {
    Header,
    Footer
    
  },
  data(){
    return {
      fill_bg:(this.helper.getUserInfo().userId > 0?'':'full-bg'),
      api_data:[]
    }
  },
  mounted(){
  }
}
</script>
