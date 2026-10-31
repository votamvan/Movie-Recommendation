<template>

  <div :class="fill_bg">
    <SiteHeader/>
    <section class="main-body"><router-view/></section>
    <SiteFooter/>
  </div>
</template>

<script>

import SiteHeader from './components/Header.vue'
import SiteFooter from './components/Footer.vue'

export default {
  name: 'App',
  components: {
    SiteHeader,
    SiteFooter
    
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
