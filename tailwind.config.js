import tailwindConfig from '@femm/tailwind-config'

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./*.{js,ts,md,vue}', './documents/**/*.{js,ts,md,vue}', './documents/**/**/*.{js,ts,md,vue}'],
  theme: {
    extend: {},
  },
  presets: [tailwindConfig],
  plugins: [],
}