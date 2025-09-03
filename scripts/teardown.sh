#!/usr/bin/env bash
set -euo pipefail
source .env
NS="$NAMESPACE"
microk8s kubectl -n $NS delete deploy --all --ignore-not-found
microk8s kubectl -n $NS delete svc dashboard --ignore-not-found
microk8s kubectl -n $NS delete pvc urbansound-data --ignore-not-found
microk8s kubectl delete pv urbansound-pv --ignore-not-found
microk8s kubectl delete ns $NS --ignore-not-found

