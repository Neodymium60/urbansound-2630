#!/usr/bin/env bash
set -euo pipefail
source .env
REC="$REGISTRY/urbannoise-rec:$TAG"
CLS="$REGISTRY/urbannoise-cls:$TAG"
DASH="$REGISTRY/urbannoise-dash:$TAG"

docker buildx build --platform linux/arm64 -t "$REC"  ./recorder  --push --progress=plain
docker buildx build --platform linux/arm64 -t "$CLS"  ./classifier --push --progress=plain
docker buildx build --platform linux/arm64 -t "$DASH" ./dashboard --push --progress=plain

echo "Verifying tags in registry..."
curl -s http://localhost:32000/v2/_catalog
curl -s http://localhost:32000/v2/urbannoise-rec/tags/list
curl -s http://localhost:32000/v2/urbannoise-cls/tags/list
curl -s http://localhost:32000/v2/urbannoise-dash/tags/list

