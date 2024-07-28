import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "My Awesome Project",
  description: "我自己的博客",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Examples', link: '/markdown-examples' }
    ],

    sidebar: [
      {
        text: 'Examples',
        items: [
          { text: 'Markdown Examples左侧', link: '/markdown-examples' },
          { text: 'Runtime API Examples左侧', link: '/api-examples' }
        ]
      }
    ],

    //修改页脚
    docFooter: {
      prev: '上一页',
      next: '下一页'
    },
    lastUpdated:{
      text: '最后更改时间',
      //自定义时间的格式
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'short'
      }
    },
    search: {
      provider: 'local'
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
    ]
  }
})
