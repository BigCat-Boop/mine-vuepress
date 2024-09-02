import DefaultTheme from "vitepress/theme";
import type { Theme } from "vitepress";
import './styles/index.scss'

export default <Theme> {
    ...DefaultTheme,
    enhanceApp({ app }) {
        // app is the VuePress App instance
        // ... 注册全局组件app 就是vue的app
    }
}