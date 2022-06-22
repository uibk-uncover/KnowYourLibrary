#!/bin/bash

# # boss
# rm -rf boss_checksums.sha256
# for f in BOSS_tiles/*.png; do
#     printf "$f:"
#     sha1sum $f >> boss_checksums.sha256
#     echo " done"
# done


# alaska
rm -rf alaska_checksums.sha256
for f in ALASKA_v2_TIFF_256_COLOR/*.tif; do
    printf "$f:"
    sha1sum $f >> alaska_checksums.sha256
    echo " done"
done

