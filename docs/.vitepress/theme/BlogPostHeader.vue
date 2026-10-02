<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import BlogAuthors from './BlogAuthors.vue'

const { page, frontmatter, lang } = useData()
const isChinese = computed(() => lang.value.startsWith('zh'))
const isBlogPost = computed(
  () => /^(en|zh)\/blog\/.+\.md$/.test(page.value.relativePath) &&
    !/\/blog\/(?:index\.md$|page\/)/.test(page.value.relativePath)
)
const date = computed(() => new Date(frontmatter.value.date).toISOString().slice(0, 10))
const authors = computed(() => [frontmatter.value.author, ...(frontmatter.value.co_authors ?? [])])
</script>

<template>
  <header v-if="isBlogPost" class="blog-post-header">
    <h1>{{ frontmatter.title }}</h1>
    <div class="publication">
      <time :datetime="date">{{ date }}</time>
      <a class="back-link" :href="withBase(isChinese ? '/zh/blog/' : '/en/blog/')">
        <span aria-hidden="true">←</span> {{ isChinese ? '所有文章' : 'All articles' }}
      </a>
    </div>
    <BlogAuthors :authors="authors" />
  </header>
</template>

<style scoped>
.blog-post-header {
  padding-bottom: 28px;
  margin-bottom: 32px;
  border-bottom: 1px solid var(--vp-c-divider);
  font-size: 14px;
}

.blog-post-header h1 {
  margin: 0;
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-headline);
  font-size: clamp(32px, 4vw, 48px);
  font-weight: 700;
  line-height: 1.25;
  letter-spacing: -0.035em;
  overflow-wrap: anywhere;
  text-wrap: balance;
}

.back-link {
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

.publication {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin: 24px 0;
  color: var(--vp-c-text-2);
}
</style>
