#!/usr/bin/env bash
# Must run from the root of the Git repo.

if [ -z "$1" ]; then
    echo "Usage: $0 [project-hooks-folder]"
    exit 1
fi

major=$(git --version | grep -o '[0-9.]*' | awk -F \. {'print $1'})
minor=$(git --version | grep -o '[0-9.]*' | awk -F \. {'print $2'})

if [ $major -eq "2" ] && [ $minor -lt "9" ]; then
    rm -rf .git/hooks/*
    find $1 -type f -exec ln -sf ../../{} .git/hooks/ \;
else
    git config core.hooksPath $1
fi

echo "Done."
