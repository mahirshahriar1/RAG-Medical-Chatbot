#!/bin/sh
set -eu

dockerd \
  --host=unix:///var/run/docker.sock \
  --storage-driver=overlay2 \
  >/var/log/dockerd.log 2>&1 &

echo "Starting Docker daemon..."
i=0
until docker info >/dev/null 2>&1; do
  i=$((i+1))
  if [ "$i" -gt 60 ]; then
    echo "Docker daemon failed to start"
    cat /var/log/dockerd.log || true
    exit 1
  fi
  sleep 1
done

echo "Docker daemon is ready"
exec su -s /bin/bash jenkins -c "/usr/local/bin/jenkins.sh"