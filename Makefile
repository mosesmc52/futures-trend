PROJECT_NAME ?= futures-trend
TAG ?= latest
IMAGE ?= ghcr.io/mosesmc52/$(PROJECT_NAME):$(TAG)

COMPOSE_FILE ?= docker/docker-compose.local.yml
COMPOSE_SERVICE ?= algo
DOCKER ?= docker
DOCKER_COMPOSE ?= $(DOCKER) compose

DO_FN_DIR ?= infra/do-functions
DO_FN_ENV ?= $(DO_FN_DIR)/.env
DO_FN_NAME ?= launcher/futures-trend

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
	@echo "  do-fn-validate       Validate DO Functions project metadata"
	@echo "  do-fn-connect        Connect doctl to a DO Functions namespace"
	@echo "  do-fn-status         Show DO Functions connection status"
	@echo "  do-fn-deploy         Deploy DO Functions with runtime env"
	@echo "  do-fn-deploy-remote  Deploy DO Functions using remote build"
	@echo "  do-fn-list           List deployed DO functions"
	@echo "  do-fn-get            Show deployed function metadata"
	@echo "  do-fn-invoke         Invoke $(DO_FN_NAME)"
	@echo "  do-fn-activations    List recent activations"
	@echo "  do-fn-logs           Show activation logs with ACTIVATION=<id>"
	@echo "  do-droplet-log       Tail droplet log with DROPLET_IP=<ip>"
	@echo "  do-spaces-log        Download a Spaces log with LOG_KEY=<key>"

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

do-fn-validate:
	doctl serverless get-metadata $(DO_FN_DIR)

do-fn-connect:
	doctl serverless connect

do-fn-status:
	doctl serverless status

do-fn-deploy:
	doctl serverless deploy $(DO_FN_DIR) --env $(DO_FN_ENV)

do-fn-deploy-remote:
	doctl serverless deploy $(DO_FN_DIR) --env $(DO_FN_ENV) --remote-build

do-fn-list:
	doctl serverless functions list

do-fn-get:
	doctl serverless functions get $(DO_FN_NAME)

do-fn-invoke:
	doctl serverless functions invoke $(DO_FN_NAME)

do-fn-activations:
	doctl serverless activations list

do-fn-logs:
	test -n "$(ACTIVATION)" || (echo "Set ACTIVATION=<id>" && exit 1)
	doctl serverless activations logs $(ACTIVATION)

do-droplet-log:
	test -n "$(DROPLET_IP)" || (echo "Set DROPLET_IP=<ip>" && exit 1)
	ssh $(DROPLET_USER)@$(DROPLET_IP) "sudo tail -f $(DROPLET_LOG_FILE)"

do-spaces-log:
	test -n "$(LOG_KEY)" || (echo "Set LOG_KEY=<spaces log key>" && exit 1)
	test -n "$(SPACES_ENDPOINT)" || (echo "Set SPACES_ENDPOINT=<https://...>" && exit 1)
	test -n "$(SPACES_BUCKET)" || (echo "Set SPACES_BUCKET=<bucket>" && exit 1)
	aws --endpoint-url $(SPACES_ENDPOINT) s3 cp s3://$(SPACES_BUCKET)/$(LOG_KEY) -
