#!/bin/sh
set -eu
cd "$(dirname "$0")"
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @long.json
