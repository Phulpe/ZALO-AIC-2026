#!/usr/bin/env bash
# Sao luu checkpoint LoRA tu may thue ve may local (chay tren Windows Git Bash).
# Dung: bash sync_ckpt.sh <port> <host> <remote_run_dir> <local_dir> [interval_s]
# - chi lay checkpoint da ghi xong (co trainer_state.json), dong goi tgz + sha256 tren may thue,
#   tai ve, kiem sha256, giai nen vao <local_dir>/checkpoints/; khong xoa gi tren may thue.
# - moi vong cung tai log/progress/metadata; dung khi thay training_complete.json da duoc dong bo.
set -u
PORT=$1; HOST=$2; RDIR=$3; LDIR=$4; INT=${5:-300}
K=/d/ZAIC/PRIVATE_DATA/SSH_VAST
O=(-i "$K/vast_ed25519" -o UserKnownHostsFile="$K/known_hosts" -o ConnectTimeout=20 -o ServerAliveInterval=30)
mkdir -p "$LDIR/checkpoints" "$LDIR/archives"
log() { echo "[sync $(date -u +%H:%M:%SZ)] $*" | tee -a "$LDIR/SYNC_LOG.txt"; }
while true; do
  scp -q "${O[@]}" -P "$PORT" "root@$HOST:$RDIR/{progress.json,metadata.json,training_complete.json,STOPPED_AT_DEADLINE}" "$LDIR/" 2>/dev/null
  scp -q "${O[@]}" -P "$PORT" "root@$HOST:$(dirname "$RDIR")/*.log" "$LDIR/" 2>/dev/null
  CKS=$(timeout 120 ssh "${O[@]}" -p "$PORT" "root@$HOST" "cd $RDIR/checkpoints 2>/dev/null && for d in checkpoint-* ; do [ -f \$d/trainer_state.json ] && echo \$d; done" 2>/dev/null | grep '^checkpoint-')
  for c in $CKS; do
    [ -f "$LDIR/checkpoints/$c/.verified" ] && continue
    if timeout 600 ssh "${O[@]}" -p "$PORT" "root@$HOST" "cd $RDIR/checkpoints && tar cf /tmp/$c.tar $c && sha256sum /tmp/$c.tar | cut -c1-64" > "$LDIR/archives/$c.sha_remote" 2>/dev/null \
       && timeout 1800 scp -q "${O[@]}" -P "$PORT" "root@$HOST:/tmp/$c.tar" "$LDIR/archives/$c.tar"; then
      R=$(tr -d '[:space:]' < "$LDIR/archives/$c.sha_remote"); L=$(sha256sum "$LDIR/archives/$c.tar" | cut -c1-64)
      if [ "$R" = "$L" ] && [ -n "$R" ]; then
        tar xf "$LDIR/archives/$c.tar" -C "$LDIR/checkpoints" && echo "$L" > "$LDIR/checkpoints/$c/.verified"
        (cd "$LDIR/checkpoints/$c" && find . -type f ! -name .verified -exec sha256sum {} \; > "../$c.FILES_SHA256.txt")
        log "OK $c sha256=$L"
        timeout 60 ssh "${O[@]}" -p "$PORT" "root@$HOST" "rm -f /tmp/$c.tar" 2>/dev/null
      else
        log "SHA MISMATCH $c remote=$R local=$L (se thu lai)"; rm -f "$LDIR/archives/$c.tar"
      fi
    else
      log "tai loi $c (se thu lai)"
    fi
  done
  if [ -f "$LDIR/training_complete.json" ]; then
    FIN=$(timeout 120 ssh "${O[@]}" -p "$PORT" "root@$HOST" "cd $RDIR && [ -f final/adapter_model.safetensors ] && tar cf /tmp/final.tar final && sha256sum /tmp/final.tar | cut -c1-64" 2>/dev/null)
    if [ -n "$FIN" ] && scp -q "${O[@]}" -P "$PORT" "root@$HOST:/tmp/final.tar" "$LDIR/archives/final.tar" && [ "$(sha256sum "$LDIR/archives/final.tar" | cut -c1-64)" = "$FIN" ]; then
      tar xf "$LDIR/archives/final.tar" -C "$LDIR" && (cd "$LDIR/final" && sha256sum * > ../final.FILES_SHA256.txt)
      log "OK final sha256=$FIN -> dung"; break
    fi
    log "final chua san sang/tai loi (se thu lai)"
  fi
  sleep "$INT"
done
