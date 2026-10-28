#!/bin/sh
# mashbox max lifetime, set by `mashbox tpu up --max-hours`. A startup script: it runs as root at every boot of the chip.
# The first boot writes the deadline (boot time + mashbox-max-seconds) to /var/lib/mashbox/deadline; every boot arms a
# timer for it; at the deadline the chip deletes itself through the Cloud TPU API with its own service account, so it
# works with the machine that ran `mashbox tpu up` switched off. Log: /var/log/mashbox-life.log on the chip.
M=http://metadata.google.internal/computeMetadata/v1
md() { curl -fsS -m 10 -H 'Metadata-Flavor: Google' "$M/$1"; }
log() { echo "$(date -u +%FT%TZ) $*" >> /var/log/mashbox-life.log; }
if [ "${1:-}" != fire ]; then
  secs=$(md instance/attributes/mashbox-max-seconds) || secs=0
  case "$secs" in ''|*[!0-9]*) secs=0 ;; esac
  [ "$secs" -gt 0 ] || { log "off (mashbox-max-seconds=$secs)"; exit 0; }
  mkdir -p /var/lib/mashbox
  [ -s /var/lib/mashbox/deadline ] || echo $(( $(date +%s) + secs )) > /var/lib/mashbox/deadline
  md instance/attributes/startup-script > /usr/local/sbin/mashbox-life && chmod 755 /usr/local/sbin/mashbox-life
  dl=$(cat /var/lib/mashbox/deadline); left=$(( dl - $(date +%s) ))   # after the fetch: a past calendar time never fires
  # A wall-clock timer: a monotonic one (--on-active) fired 5 min late on a queued chip (10-28), most likely because its
  # VM was paused while Google provisioned it (the monotonic clock stops, the wall clock does not).
  if [ "$left" -le 2 ]; then nohup /usr/local/sbin/mashbox-life fire >/dev/null 2>&1 & how="now, deadline passed"
  elif systemd-run --unit=mashbox-life --timer-property=AccuracySec=1s \
       --on-calendar="$(date -u -d @"$dl" '+%Y-%m-%d %H:%M:%S') UTC" /usr/local/sbin/mashbox-life fire >/dev/null 2>&1; then
    how=systemd-run
  else nohup sh -c "while [ \$(date +%s) -lt $dl ]; do sleep 20; done; /usr/local/sbin/mashbox-life fire" >/dev/null 2>&1 &
    how="wall-clock loop"; fi
  log "armed ($how): delete at $(date -u -d @"$dl" +%FT%TZ), in ${left}s"; exit 0
fi
p=$(md instance/attributes/mashbox-project); z=$(md instance/attributes/mashbox-zone)
n=$(md instance/attributes/mashbox-node); q=$(md instance/attributes/mashbox-qr || true)
api="https://tpu.googleapis.com/v2/projects/$p/locations/$z"
if [ -n "$q" ]; then url="$api/queuedResources/$q?force=true"; else url="$api/nodes/$n"; fi
umask 077; h=/run/mashbox-life.h
for i in 1 2 3 4 5 6 7 8 9 10; do
  md instance/service-accounts/default/token | sed -n 's/.*"access_token" *: *"\([^"]*\)".*/Authorization: Bearer \1/p' > "$h"
  code=$(curl -s -m 60 -o /run/mashbox-life.out -w '%{http_code}' -X DELETE -H @"$h" "$url"); rm -f "$h"
  log "try $i: DELETE $url: HTTP $code"; [ "$code" = 200 ] && exit 0
  sleep 60
done
