#!/usr/bin/env bash
set -euo pipefail
# BuildKit config so pushing to localhost:32000 (HTTP) works
cat > buildkitd.toml <<'EOF'
[registry."localhost:32000"]
  http = true
  insecure = true
EOF
docker buildx rm arm64builder >/dev/null 2>&1 || true
docker buildx create --name arm64builder --driver docker-container \
  --driver-opt network=host --config ./buildkitd.toml --use
docker run --privileged --rm tonistiigi/binfmt --install all
docker buildx inspect --bootstrap
echo "Builder ready."

