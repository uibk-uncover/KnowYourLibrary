#!/bin/bash

# clean
rm -rf boss_checksums.sha256

# compute hash of each file
for f in BOSS_tiles/*.png; do
    printf "$f:"
    sha1sum $f >> boss_checksums.sha256
    echo " done"
done

