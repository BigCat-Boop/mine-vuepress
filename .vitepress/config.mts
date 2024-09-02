import { defineConfig } from "vitepress";
import path from "path";
import autoGetSidebarOptionBySrcDir from "../sidebar.ts";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "maomaoliao",
  description: "我自己的博客",
  head: [
    [
      "link",
      {
        rel: "icon",
        type: "image/webp",
        href: "/cat.webp",
      },
    ],
    [
      "meta",
      {
        name: "author",
        content: "liaoyan",
      },
    ],
  ],
  outDir: "docs", //打包输出的目录 把打包的docs输出到外面 这个是固定写死的(打包生成工具也得写死)\
  // base: '/mine-vuepress/',
  appearance: "dark",
  themeConfig: {
    // repo: "clark-cui/homeSite",
    logo: "/cat.png",
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: "🏡主页", link: "/" },
      // { text: "📃简历", link: "/documents/about_me/我的简历" },
      {
        text: "🔖前端技术",
        items: [
          { text: "🖥️ 埋点监控", link: "/documents/font_end/埋点监控" },
          {
            text: "🔥 webpack打包源码",
            link: "/documents/font_end/webpack打包源码",
          },
          {
            text: "📈 ccid自动化部署",
            link: "/documents/font_end/ccid自动化部署",
          },
          {
            text: "👌 pinia手写源码",
            link: "/documents/font_end/pinia手写源码",
          },
        ],
      },
      {
        text: "🔥AI技术",
        items: [
          { text: "🔥NLP", link: "/documents/artificial/NLP/Transformer" },
          {
            items: [
              {
                text: "🤖 Transformer",
                link: "/documents/artificial/NLP/Transformer",
              },
            ],
          },
          { text: "🔥CNN", link: "/documents/artificial/CNN/Yolov5自我解析" },
          {
            items: [
              {
                text: "🌱 Yolov5自我解析",
                link: "/documents/artificial/CNN/Yolov5自我解析",
              },
            ],
          },
        ],
      },
    ],

    sidebar: {
      "/documents/about_me/": autoGetSidebarOptionBySrcDir(
        path.resolve(__dirname, "../documents/about_me"),
        "个人资料"
      ),
      "/documents/artificial/": autoGetSidebarOptionBySrcDir(
        path.resolve(__dirname, "../documents/artificial/NLP"),
        "NLP"
      ).concat(
        autoGetSidebarOptionBySrcDir(
          path.resolve(__dirname, "../documents/artificial/CNN"),
          "CNN"
        )
      ),
      "/documents/font_end": autoGetSidebarOptionBySrcDir(
        path.resolve(__dirname, "../documents/font_end"),
        "前端技术"
      ),
    },

    //修改页脚
    docFooter: {
      prev: "上一页",
      next: "下一页",
    },
    lastUpdated: {
      text: "最后更改时间",
      //自定义时间的格式
      formatOptions: {
        dateStyle: "full",
        timeStyle: "short",
      },
    },
    search: {
      provider: "local",
    },

    // socialLinks: [
    //   { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
    // ]
  },
});
