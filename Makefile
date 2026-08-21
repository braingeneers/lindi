PYTHON_VERSION ?= 3.12
TEST_IMAGE ?= lindi-tests:py$(PYTHON_VERSION)
DOCKER_TEST_RUN = docker run --rm --user "$(shell id -u):$(shell id -g)" -v "$(CURDIR):/workspace" -w /workspace $(TEST_IMAGE)

.PHONY: test test-local test-docker test-network-docker test-live-object-store-docker

test: test-docker

test-local:
	pytest -m "not network and not live_object_store" tests/

test-docker:
	docker build --build-arg PYTHON_VERSION=$(PYTHON_VERSION) -f Dockerfile.test -t $(TEST_IMAGE) .
	$(DOCKER_TEST_RUN) pytest -m "not network and not live_object_store" --cov lindi --cov-branch --cov-report=xml --cov-report=term tests/

test-network-docker:
	docker build --build-arg PYTHON_VERSION=$(PYTHON_VERSION) -f Dockerfile.test -t $(TEST_IMAGE) .
	$(DOCKER_TEST_RUN) pytest -m network tests/

test-live-object-store-docker:
	docker build --build-arg PYTHON_VERSION=$(PYTHON_VERSION) -f Dockerfile.test -t $(TEST_IMAGE) .
	docker run --rm --user "$(shell id -u):$(shell id -g)" -e LINDI_LIVE_REFERENCE_URL -e LINDI_LIVE_DATASET_PATH -v "$(CURDIR):/workspace" -w /workspace $(TEST_IMAGE) pytest -m live_object_store tests/test_live_object_store.py
