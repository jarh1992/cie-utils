#!/bin/bash

# General-purpose script to publish Sphinx documentation to the `gh-pages` branch.
# Can be used by any Python project with docs in `docs/` and a GitHub repo.

set -e  # Stop on error

# --------- CONFIGURABLE VARIABLES ---------
BUILD_DIR="docs/_build/html"
REPO_URL=$(git config --get remote.origin.url)
GH_PAGES_BRANCH="gh-pages"
TMP_DIR=$(mktemp -d)
COMMIT_MESSAGE="📚 Update Sphinx documentation"
# ------------------------------------------

echo "🔧 Building documentation with Sphinx..."
sphinx-build -b html docs "$BUILD_DIR"

echo "📁 Cloning the repo's $GH_PAGES_BRANCH branch into temporary directory..."
git clone --branch "$GH_PAGES_BRANCH" "$REPO_URL" "$TMP_DIR"

echo "🧹 Cleaning existing files..."
rm -rf "$TMP_DIR"/*

echo "📤 Copying new documentation into the temporary repo..."
cp -r "$BUILD_DIR"/* "$TMP_DIR"

cd "$TMP_DIR"
git config user.name "${GIT_AUTHOR_NAME:-autodoc bot}"
git config user.email "${GIT_AUTHOR_EMAIL:-autodoc@localhost}"

echo "✅ Committing and pushing changes..."
git add .
git commit -m "$COMMIT_MESSAGE"
git push origin "$GH_PAGES_BRANCH"

echo "✅ Documentation has been successfully published to '$GH_PAGES_BRANCH'!"
