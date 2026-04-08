PROJECT_NAME ?= futures-trend
TAG ?= latest
IMAGE ?= ghcr.io/mosesmc52/$(PROJECT_NAME):$(TAG)

COMPOSE_FILE ?= docker/docker-compose.local.yml
COMPOSE_SERVICE ?= algo
DOCKER ?= docker
DOCKER_COMPOSE ?= $(DOCKER) compose

DROPLET_USER ?= root
DROPLET_LOG_FILE ?= /var/log/job.log
SPACES_ENDPOINT ?=
SPACES_BUCKET ?=

.PHONY: help build up upd shell logs restart stop down clean ps \
	image-build image-push \
	do-fn-validate do-fn-connect do-fn-status do-fn-deploy do-fn-deploy-remote \
	do-fn-list do-fn-get do-fn-invoke do-fn-activations do-fn-logs \
	do-droplet-log do-spaces-log

help:
	@echo "Available targets:"
	@echo "  build                Build the local Docker compose service"
	@echo "  up                   Start the app in the foreground"
	@echo "  upd                  Start the app in detached mode"
	@echo "  shell                Open a shell in the running app container"
	@echo "  logs                 Tail container logs"
	@echo "  ps                   Show compose service status"
	@echo "  restart              Restart the Docker service"
	@echo "  stop                 Stop the Docker service"
	@echo "  down                 Stop and remove the Docker service"
	@echo "  clean                Stop compose and remove the built image"
	@echo "  image-build          Build the GHCR image tag: $(IMAGE)"
	@echo "  image-push           Push the GHCR image tag: $(IMAGE)"

build:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) build $(COMPOSE_SERVICE)

up:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up $(COMPOSE_SERVICE)

upd:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up -d $(COMPOSE_SERVICE)

shell:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) exec $(COMPOSE_SERVICE) /bin/bash

logs:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f $(COMPOSE_SERVICE)

ps:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) ps

restart:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) restart $(COMPOSE_SERVICE)

stop:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) stop $(COMPOSE_SERVICE)

down:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down

clean: down
	-$(DOCKER) image rm $(IMAGE)

image-build:
	$(DOCKER) build -f docker/Dockerfile -t $(IMAGE) .

image-push:
	$(DOCKER) push $(IMAGE)
