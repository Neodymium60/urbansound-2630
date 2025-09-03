#!/usr/bin/env bash
set -euo pipefail
source .env

# derive & export once
export REC="$REGISTRY/urbannoise-rec:$TAG"
export CLS="$REGISTRY/urbannoise-cls:$TAG"
export DASH="$REGISTRY/urbannoise-dash:$TAG"
export NODE_NAME NAMESPACE MIC_DEVICE USE_TPU USB_MOUNT PVC_SIZE

# apply PV/PVC as you already do …

# apply deploys (each gets the same environment)
envsubst < k8s/10-deploy-recorder.yaml   | microk8s kubectl apply -f -
envsubst < k8s/11-deploy-classifier.yaml | microk8s kubectl apply -f -
envsubst < k8s/12-deploy-dashboard.yaml  | microk8s kubectl apply -f -


# Namespace
microk8s kubectl apply -f k8s/00-namespace.yaml

# Node label/taint (idempotent)
N="$NODE_NAME" microk8s kubectl -n "$NAMESPACE" apply -f k8s/03-config-asound.yaml
N="$NODE_NAME" bash k8s/01-node-label-taint.sh

# Ensure USB mount exists on the Pi node (run once on the Pi manually):
#   sudo mkdir -p "$USB_MOUNT" && sudo chmod 0777 "$USB_MOUNT"
# Then create PV/PVC bound to that path/node
USB_PATH="$USB_MOUNT" PVC_SIZE="$PVC_SIZE" envsubst < k8s/02-pv-pvc-usb.yaml | microk8s kubectl apply -f -

# Deploy workloads (images use $REGISTRY and $TAG)
REC="$REGISTRY/urbannoise-rec:$TAG" \
CLS="$REGISTRY/urbannoise-cls:$TAG" \
DASH="$REGISTRY/urbannoise-dash:$TAG" \
MIC_DEVICE="$MIC_DEVICE" USE_TPU="$USE_TPU" \
envsubst < k8s/10-deploy-recorder.yaml   | microk8s kubectl apply -f -
envsubst < k8s/11-deploy-classifier.yaml | microk8s kubectl apply -f -
envsubst < k8s/12-deploy-dashboard.yaml  | microk8s kubectl apply -f -
microk8s kubectl apply -f k8s/13-svc-dashboard.yaml

echo "Set MIC device:"
microk8s kubectl -n "$NAMESPACE" set env deploy/recorder MIC_DEVICE="$MIC_DEVICE"

