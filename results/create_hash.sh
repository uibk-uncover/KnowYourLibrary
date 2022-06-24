#!/bin/bash

# boss
rm -rf compression.sha256
for f in compression/*/*.jpeg; do
    printf "$f:"
    sha1sum $f >> compression.sha256
    echo " done"
done


# alaska
rm -rf decompression.sha256
for f in decompression/*/*.png; do
    printf "$f:"
    sha1sum $f >> decompression.sha256
    echo " done"
done

