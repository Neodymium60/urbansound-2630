#!/usr/bin/env bash
set -euo pipefail
: "${N:=pi-ndr}"
# Label to select the Pi
microk8s kubectl label node "$N" urbansound-node="$N" --overwrite
# Optional: taint so ONLY urbansound pods land here
microk8s kubectl taint nodes "$N" urbansound=only:NoSchedule --overwrite || true

