#!/usr/bin/env node

const fs = require('node:fs')
const path = require('node:path')

const imageDir = path.join(__dirname, 'public/images')
const maxBytes = 300 * 1024

if (!fs.existsSync(path.join(imageDir, '.git'))) {
  console.error('Initialize images with: git submodule update --init docs/public/images')
  process.exit(1)
}

let oversized = 0
const checkDirectory = (directory) => {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    if (entry.name === '.git') continue
    const filename = path.join(directory, entry.name)
    if (entry.isDirectory()) {
      checkDirectory(filename)
    } else {
      const size = fs.statSync(filename).size
      if (size > maxBytes) {
        console.error(`${path.relative(imageDir, filename)}: ${size} bytes exceeds 300 KiB`)
        oversized++
      }
    }
  }
}

checkDirectory(imageDir)
if (oversized) {
  console.error('Compress or split oversized image assets before updating the image submodule.')
  process.exitCode = 1
} else {
  console.log('Image submodule files are within the 300 KiB limit.')
}
