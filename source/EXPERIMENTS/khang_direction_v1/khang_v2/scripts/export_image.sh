#!/usr/bin/env bash
# Xuat image, nen, tao MD5 theo muc 11 huong dan Docker cua BTC.
set -euo pipefail
set -o pipefail
TEAM="${1:?ten doi}"
TAG="zaic2026_datamate_${TEAM}:final"
OUT="zaic2026_datamate_${TEAM}.tar.gz"
if command -v pigz >/dev/null; then Z="pigz -p 8"; UZ="pigz -dc"; else Z="gzip"; UZ="gzip -dc"; fi
docker save "${TAG}" | ${Z} > "${OUT}"
SIZE=$(${UZ} "${OUT}" | wc -c)
md5sum "${OUT}" > "${OUT}.md5"
echo "File: ${OUT}  ($(du -h "${OUT}" | cut -f1) nen)"
echo "Kich thuoc .tar chua nen (byte): ${SIZE}  (gioi han BTC: 100 GB)"
cat "${OUT}.md5"
