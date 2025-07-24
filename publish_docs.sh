#!/bin/bash

# ------------------------------------------------------------------------------
# General-purpose script to publish Sphinx documentation to the `gh-pages` branch
# Uses git worktree instead of cloning (safer and faster)
# ------------------------------------------------------------------------------
set -e  # Abort on error

# -------- CONFIGURATION --------
BUILD_DIR="docs/build/html"
SOURCE_DIR="docs/source"
GH_PAGES_BRANCH="gh-pages"
WORKTREE_DIR=".gh-pages-tmp"
COMMIT_MESSAGE="📚 Update Sphinx documentation"
# -------------------------------

echo "🔧 Building Sphinx documentation..."
sphinx-build -b html "$SOURCE_DIR" "$BUILD_DIR"

echo "🌿 Preparing git worktree for '$GH_PAGES_BRANCH' branch..."
rm -rf "$WORKTREE_DIR"
git worktree add "$WORKTREE_DIR" "$GH_PAGES_BRANCH"

echo "🧹 Cleaning old documentation..."
rm -rf "$WORKTREE_DIR"/*
cp -r "$BUILD_DIR"/* "$WORKTREE_DIR"

echo "➕ Adding .nojekyll for GitHub Pages..."
touch "$WORKTREE_DIR/.nojekyll"

cd "$WORKTREE_DIR"
git config user.name "${GIT_AUTHOR_NAME:-autodoc bot}"
git config user.email "${GIT_AUTHOR_EMAIL:-autodoc@localhost}"

echo "📦 Committing and pushing changes..."
git add .
git commit -m "$COMMIT_MESSAGE"
git push origin "$GH_PAGES_BRANCH"

cd ..
echo "🧹 Cleaning up worktree..."
git worktree remove "$WORKTREE_DIR" --force

echo "✅ Documentation successfully published to '$GH_PAGES_BRANCH'!"
