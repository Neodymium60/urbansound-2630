#!/usr/bin/env bash
set -euo pipefail
NS=${NAMESPACE:-urbansound}
NODE=${NODE_NAME:-pi-ndr}

# Ensure NS exists
microk8s kubectl get ns "$NS" >/dev/null 2>&1 || microk8s kubectl create ns "$NS"

# Label/taint (ok to keep here)
microk8s kubectl label node "$NODE" urbansound-node="$NODE" --overwrite
microk8s kubectl taint nodes "$NODE" urbansound=only:NoSchedule --overwrite

# ---- STATUS ONLY (no applies below this line) ----
echo "PV/PVC:"
microk8s kubectl get pv urbansound-pv -o wide || true
microk8s kubectl -n "$NS" get pvc urbansound-data -o wide || true

echo "Deployments:"
microk8s kubectl -n "$NS" get deploy \
  -o custom-columns=NAME:.metadata.name,IMAGE:.spec.template.spec.containers[*].image

echo "Rollout status:"
microk8s kubectl -n "$NS" rollout status deploy/recorder    || true
microk8s kubectl -n "$NS" rollout status deploy/classifier  || true
microk8s kubectl -n "$NS" rollout status deploy/dashboard   || true

echo "Pods:"
microk8s kubectl -n "$NS" get pods -o wide

