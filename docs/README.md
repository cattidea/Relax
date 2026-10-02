# Relax Documentation

This directory contains the VitePress documentation site for Relax.

## Features

- 📚 Bilingual documentation (English & Chinese)
- 🎨 Beautiful VitePress theme with custom branding
- 🔍 Full-text search support
- 🖼️ **Image zoom functionality** - Click any image to view it in full size
- 📊 Mermaid diagram support
- 🌓 Dark mode support

## Development

```bash
# Install dependencies
npm install

# Start dev server
npm run docs:dev

# Build for production
npm run docs:build

# Preview production build
npm run docs:preview
```

## Structure

```
docs/
├── .vitepress/
│   ├── config.mts          # VitePress configuration
│   └── theme/              # Custom theme
├── public/                 # Static assets
├── en/                     # English documentation
│   ├── guide/              # English guides
│   │   ├── introduction.md
│   │   ├── installation.md
│   │   ├── quick-start.md
│   │   └── ...
│   ├── api/                # API documentation
│   ├── examples/           # Example documentation
│   └── index.md            # English homepage
├── zh/                     # Chinese documentation
│   ├── guide/              # Chinese guides
│   ├── api/                # API documentation
│   ├── examples/           # Example documentation
│   └── index.md            # Chinese homepage
├── draft/                  # Draft documentation
│   ├── design.md
│   ├── metrics_service_usage.md
│   └── ...
└── index.md                # Root homepage (defaults to Chinese)
```

## Adding New Pages

1. Create a new markdown file in the appropriate directory
2. Add the page to the sidebar in `.vitepress/config.mts`
3. Add translations if needed

## Publishing Blog Posts / 发布博客

The Blog navigation opens the article list. Its Markdown loader follows
[PFCCLab's implementation](https://github.com/PFCCLab/blog/tree/main/src/.vitepress/theme/loaders),
with a Relax-themed layout informed by the [Vue](https://blog.vuejs.org/),
[Hugging Face](https://huggingface.co/blog), and [PyTorch](https://pytorch.org/blog/) blogs.
Add matching Markdown files at `en/blog/<slug>.md` and `zh/blog/<slug>.md` using the
frontmatter below. Translate the title, description, and body in each version.
The list automatically displays each language's posts, newest first; no sidebar
or navigation edits are needed for individual posts.
Each language has 10 posts per page. Pagination appears when there are more than
10 posts; later pages use URLs such as `/en/blog/page/2.html`. Like PFCCLab, the
site generates separate HTML pages during the build, so direct links and refreshes work.

博客导航进入文章列表，Markdown 加载方式参考
[PFCCLab 源码](https://github.com/PFCCLab/blog/tree/main/src/.vitepress/theme/loaders)，
布局借鉴 [Vue](https://blog.vuejs.org/)、[Hugging Face](https://huggingface.co/blog) 和
[PyTorch](https://pytorch.org/blog/) 博客，配色与字体沿用 Relax 官网。
在 `en/blog/<slug>.md` 和 `zh/blog/<slug>.md` 添加同名的中英文文章，填写以下元数据，
并翻译标题、摘要和正文。列表会按日期倒序自动收录对应语言的文章，无需逐篇修改导航或侧边栏。
每种语言每页显示 10 篇，超过 10 篇时显示分页；后续页面使用 `/zh/blog/page/2.html` 这样的地址。
与 PFCCLab 一样，分页会在构建时生成独立 HTML，支持直接打开链接和刷新。

```markdown
---
title: Your article title
date: '2026-10-02'
author:
  name: 渡晓
  github: SigureMo
co_authors:
  - name: 禹哲
    github: NINGBENZHE
  - name: 月天
    github: Aurelius84
description: A short summary for the article list.
---

Article body.
```

`title`, `date`, `author`, and `description` are required; `co_authors` is optional.
Use `YYYY-MM-DD` for the publication date.
The page renders its heading from `title`; start the body directly, without
repeating an H1. Use H2 (`##`) for sections.
For `author`, set `name` to the display name and `github` to the GitHub username
without `@`. The article list and header display the GitHub avatar and link to
the profile. A plain author name is also supported for authors without GitHub.
Following PFCCLab's format, `co_authors` is a list with the same structure as
`author`. Authors appear in order: the primary author, then the co-authors.
Multi-author cards show avatars that overlap as space narrows, with the primary
author in front. Article headers show names and avatars in equal-width columns
that adapt to the available space.
Keep `index.md` as the listing page. Files in the blog directories are published
with the documentation site; add articles there when they are ready to be shared.
The `page/` directory is reserved for pagination templates.

`title`、`date`、`author` 和 `description` 为必填，`co_authors` 可省略。
发布日期使用 `YYYY-MM-DD` 格式。`index.md` 保留为列表页。
页面会根据 `title` 自动显示大标题，正文无需重复写一级标题；章节从二级标题（`##`）开始。
`author.name` 是显示昵称，`author.github` 是不带 `@` 的 GitHub 用户名。
列表和文章页会显示 GitHub 头像并链接到个人主页。没有 GitHub 账号的作者也可以直接填写姓名字符串。
与 PFCCLab 一样，`co_authors` 使用列表，每一项的格式与 `author` 相同。
显示顺序为主作者在前、共同作者随后。多作者卡片只显示头像，空间变窄时逐渐重叠，
主作者位于最前层；文章页用等宽网格显示头像和姓名，列数随可用宽度调整。
博客目录下的文章会随文档站一起发布，在文章可以公开时再放入该目录。
`page/` 目录保留给分页模板使用。

## Image Zoom Feature

All images in the documentation support click-to-zoom functionality powered by `medium-zoom`.

**Usage:**

- Hover over any image to see the zoom cursor
- Click to enlarge the image
- Click again or press ESC to close

**Documentation:**

- [Quick Start Guide](QUICK_START_IMAGE_ZOOM.md)
- [Feature Details](IMAGE_ZOOM_FEATURE.md)
- [Implementation Summary](IMAGE_ZOOM_IMPLEMENTATION.md)
- [Testing Guide](TESTING_IMAGE_ZOOM.md)

## Deployment

The documentation can be deployed to:

- GitHub Pages
- Vercel
- Netlify
- Any static hosting service

See [VitePress deployment guide](https://vitepress.dev/guide/deploy) for details.
