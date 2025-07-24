#!/bin/bash

set -e  # Abort on error

# -------- CONFIGURATION --------
BUILD_DIR="docs/build/html"
SOURCE_DIR="docs/source"
GH_PAGES_BRANCH="gh-pages"
WORKTREE_DIR=".gh-pages-tmp"
COMMIT_MESSAGE="📚 Update Sphinx documentation"
# -------------------------------

# Ensure _static exists to avoid warning
mkdir -p "$SOURCE_DIR/_static"

echo "🔧 Building Sphinx documentation..."
sphinx-build -b html "$SOURCE_DIR" "$BUILD_DIR"

echo "🌿 Checking if '$GH_PAGES_BRANCH' branch exists..."
if ! git show-ref --quiet refs/heads/$GH_PAGES_BRANCH; then
    echo "➕ Creating orphan branch '$GH_PAGES_BRANCH'..."
    git checkout --orphan "$GH_PAGES_BRANCH"
    git reset --hard
    touch .placeholder
    git add .placeholder
    git commit -m "Initialize $GH_PAGES_BRANCH branch"
    git push origin "$GH_PAGES_BRANCH"
    git checkout -
    git branch -D "$GH_PAGES_BRANCH"
fi

echo "🌿 Preparing git worktree for '$GH_PAGES_BRANCH' branch..."
# Clean up stale worktree
git worktree prune
rm -rf "$WORKTREE_DIR"
git worktree add "$WORKTREE_DIR" "$GH_PAGES_BRANCH"

echo "🧹 Cleaning old documentation..."
rm -rf "$WORKTREE_DIR"/*
cp -r "$BUILD_DIR"/* "$WORKTREE_DIR"
touch "$WORKTREE_DIR/.nojekyll"

cd "$WORKTREE_DIR"
git config user.name "${GIT_AUTHOR_NAME:-autodoc bot}"
git config user.email "${GIT_AUTHOR_EMAIL:-autodoc@localhost}"

echo "📦 Committing and pushing changes..."
if git status --porcelain | grep .; then
    git add .
    git commit -m "$COMMIT_MESSAGE"
    git push origin "$GH_PAGES_BRANCH"
else
    echo "✔️ No changes to commit."
fi

cd ..
echo "🧹 Cleaning up worktree..."
git worktree remove "$WORKTREE_DIR" --force

echo "✅ Documentation successfully published to '$GH_PAGES_BRANCH'!"
