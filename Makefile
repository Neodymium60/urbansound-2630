SHELL := /bin/bash

-include .env
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' .env)

NS ?= urbansound
CTRL_REGISTRY ?= 127.0.0.1:32000
IMAGE_TAG ?= 0.1.16

REC  := $(CTRL_REGISTRY)/urbannoise-rec:$(IMAGE_TAG)
CLS  := $(CTRL_REGISTRY)/urbannoise-cls:$(IMAGE_TAG)
DASH := $(CTRL_REGISTRY)/urbannoise-dash:$(IMAGE_TAG)

.PHONY: check print build set-images deploy rollout logs port-forward clean node-prune images pods events

check:
	@if [[ -z "$(CTRL_REGISTRY)" || -z "$(IMAGE_TAG)" ]]; then \
	  echo "ERROR: CTRL_REGISTRY='$(CTRL_REGISTRY)' or IMAGE_TAG='$(IMAGE_TAG)' is empty."; \
	  echo "Set them in .env or env, e.g. CTRL_REGISTRY=192.168.1.211:32000 IMAGE_TAG=0.1.16"; \
	  exit 1; \
	fi

print:
	@echo "NS            = $(NS)"
	@echo "CTRL_REGISTRY = $(CTRL_REGISTRY)"
	@echo "IMAGE_TAG     = $(IMAGE_TAG)"
	@echo "REC           = $(REC)"
	@echo "CLS           = $(CLS)"
	@echo "DASH          = $(DASH)"

build: check
	@if ! docker buildx inspect arm64builder >/dev/null 2>&1; then \
	  docker buildx create --use --name arm64builder --driver docker-container --driver-opt network=host; \
	else \
	  docker buildx use arm64builder; \
	fi
	docker buildx inspect --bootstrap
	docker buildx build --platform linux/arm64 -t "$(REC)"  ./recorder  --push
	docker buildx build --platform linux/arm64 -t "$(CLS)"  ./classifier --push
	docker buildx build --platform linux/arm64 -t "$(DASH)" ./dashboard --push

set-images: check
	microk8s kubectl -n $(NS) set image deploy/recorder   recorder=$(REC)
	microk8s kubectl -n $(NS) set image deploy/classifier classifier=$(CLS)
	microk8s kubectl -n $(NS) set image deploy/dashboard  dash=$(DASH)

deploy:
	REGISTRY=$(CTRL_REGISTRY) NS=$(NS) IMAGE_TAG=$(IMAGE_TAG) TAG=$(IMAGE_TAG) ./scripts/deploy-all.sh

rollout:
	./scripts/rollout-check.sh

logs:
	microk8s kubectl -n $(NS) logs deploy/recorder   --tail=100 || true
	microk8s kubectl -n $(NS) logs deploy/classifier --tail=100 || true
	microk8s kubectl -n $(NS) logs deploy/dashboard  --tail=100 || true

port-forward:
	microk8s kubectl -n $(NS) port-forward svc/dashboard 5000:5000

clean:
	microk8s kubectl -n $(NS) delete all --all || true
	microk8s kubectl delete ns $(NS) --ignore-not-found

node-prune:
	@echo "Pruning old urbannoise images on node (keeping tag $(IMAGE_TAG))..."
	@sudo microk8s ctr images ls -q | grep urbannoise | grep -v ":$(IMAGE_TAG)" | xargs -r -n1 sudo microk8s ctr images rm || true
	@sudo microk8s ctr snapshots ls 2>/dev/null | awk 'NR>1{print $$1}' | xargs -r -n1 sudo microk8s ctr snapshots rm || true

images:
	microk8s kubectl -n $(NS) get deploy -o custom-columns=NAME:.metadata.name,IMAGE:.spec.template.spec.containers[0].image

pods:
	microk8s kubectl -n $(NS) get pods -o wide

events:
	microk8s kubectl get events --sort-by=.lastTimestamp -A | tail -n 80

