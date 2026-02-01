#!/bin/bash

# release.sh
# Script to automate version bumping and git tagging.
# PyPI publishing and docs deployment are handled by GitHub Actions.

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Helper to print usage
print_usage() {
  echo -e "${GREEN}Usage: $0 [patch|minor|major]${NC}"
  echo -e "Increments version, commits, and tags. GitHub Actions handles PyPI publishing and docs deployment."
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

# 2. Bump version using Poetry
echo -e "${GREEN}Bumping version ($PART)...${NC}"
poetry version $PART

# 3. Extract new version
NEW_VERSION=$(poetry version -s)

echo -e "\n${GREEN}✅ Version updated to: $NEW_VERSION${NC}"

# 4. Commit and tag
git add pyproject.toml
git commit -m "🔖 Release v$NEW_VERSION"
git tag "v$NEW_VERSION"
git push
git push --tags

echo -e "\n${GREEN}🚀 Tag v$NEW_VERSION pushed! GitHub Actions will:${NC}"
echo -e "   - Run tests"
echo -e "   - Publish to PyPI"
echo -e "   - Deploy documentation to GitHub Pages"
