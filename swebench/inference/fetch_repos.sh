#!/bin/bash

mkdir repos                   # create a directory that will hold every checkout

while IFS=, read repo commit  # read each CSV line: “owner/project,<commit_sha>”
do
  owner=${repo%%/*}           # part before the slash → “owner”
  name=${repo##*/}            # part after  the slash → “project”

  mirror="https://github.com/SWE-bench-repos/${owner}__${name}.git"
  dst="/Volumes/T9/repos/${owner}__${name}__${commit}"

  git clone --quiet "$mirror" "$dst"          # fetch the mirror
  (cd "$dst" && git checkout --quiet "$commit")  # move to the correct commit
done < repos_needed.txt       # ← all stdin for the while‑loop comes from this file