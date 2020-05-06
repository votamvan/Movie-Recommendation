import Vue from 'vue'
import Router from 'vue-router'
import Home from './components/Home.vue'
import ItemDetails from './components/ItemDetails.vue'
import About from './components/About.vue'
import Cms from './components/Cms.vue'
import Login from './components/Login.vue'


Vue.use(Router)

export default new Router({
	mode: 'history',
	routes:[
		{
			path:'/',
			name:'Home',
			component:Home
		},
		{
			path:'/details/:id',
			name:'ItemDetails',
			component:ItemDetails
		},
		{
			path:'/login',
			name:'Login',
			component:Login
		},
		{
			path:'/about',
			name:'About',
			component:About
		},	
		{
			path:'/cms/:sluging',
			name:'Cms',
			component:Cms
		}
	],
	duplicateNavigationPolicy: 'reload'
}) 