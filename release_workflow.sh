#!/bin/bash

# release.sh
# Script to automate version bumping, git tagging, and PyPI publication using Poetry.

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Helper to print usage
print_usage() {
  echo -e "${GREEN}Usage: $0 [patch|minor|major]${NC}"
  echo -e "Increments version, commits, tags, builds, and uploads to PyPI using Poetry."
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

# 5. Build and publish using Poetry
echo -e "${GREEN}Building package...${NC}"
poetry build

echo -e "${GREEN}Publishing to PyPI...${NC}"
poetry publish

echo -e "\n${GREEN}🎉 v$NEW_VERSION published to PyPI successfully!${NC}"
