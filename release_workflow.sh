#!/bin/bash

# release.sh
# Script to automate version bumping, git tagging, and PyPI publication.

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Helper to print usage
print_usage() {
  echo -e "${GREEN}Usage: $0 [patch|minor|major]${NC}"
  echo -e "Increments version, commits, tags, builds, and uploads to PyPI."
}

# 1. Check argument
if [[ "$#" -ne 1 ]]; then
  print_usage
  exit 1
fi

PART=$1
if [[ "$PART" != "patch" && "$PART" != "minor" && "$PART" != "major" ]]; then
  echo -e "${RED}Invalid argument: $PART${NC}"
  print_usage
  exit 1
fi

# 2. Extract current version
CURRENT_VERSION=$(grep '^version' pyproject.toml | sed -E "s/version = \"([0-9]+)\.([0-9]+)\.([0-9]+)\"/\1.\2.\3/")
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

# 3. Bump version
case $PART in
  patch)
    PATCH=$((PATCH + 1))
    ;;
  minor)
    MINOR=$((MINOR + 1))
    PATCH=0
    ;;
  major)
    MAJOR=$((MAJOR + 1))
    MINOR=0
    PATCH=0
    ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"

# 4. Update pyproject.toml
sed -i "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" pyproject.toml

echo -e "\n${GREEN}✅ Version updated: $CURRENT_VERSION → $NEW_VERSION${NC}"

# 5. Commit and tag
git add pyproject.toml
git commit -m "🔖 Release v$NEW_VERSION"
git tag "v$NEW_VERSION"
git push
git push --tags

# 6. Build
rm -rf dist/
python -m build

# 7. Publish
python -m twine upload dist/*

echo -e "\n${GREEN}🎉 v$NEW_VERSION published to PyPI successfully!${NC}"
