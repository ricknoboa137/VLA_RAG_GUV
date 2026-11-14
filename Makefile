.PHONY: check lint format types test pins smoke lock docker-build docker-check docker-smoke

check: lint types test pins

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff check --fix src tests scripts
	ruff format src tests scripts

types:
	mypy --strict src

test:
	pytest -q -m "not sim"

pins:
	python scripts/check_pins.py

smoke:
	carma run --config configs/experiment/smoke.yaml --seed 0

# ---------------------------------------------------------------- docker
# The shareable path: a colleague needs Docker and nothing else.

docker-build:
	docker compose -f docker/compose.yaml build lib check

# Every gate, inside the image, against the locked versions.
docker-check:
	docker compose -f docker/compose.yaml run --rm check

# The smoke experiment in the container. Its manifest records the platform, so
# a result from here is distinguishable from one run on a host.
docker-smoke:
	docker compose -f docker/compose.yaml run --rm lib carma run --config configs/experiment/smoke.yaml --seed 0

# Re-resolve the locks after changing pyproject.toml. Resolved inside the image
# the locks are for, because wheels and versions differ by platform. Prints the
# lists; paste each into its lock file, keeping the header.
lock:
	docker run --rm -v "$(CURDIR):/repo:ro" python:3.11-slim-bookworm bash -c "cd /tmp && cp -r /repo/pyproject.toml /repo/README.md /repo/src /tmp/ && pip install --quiet /tmp && echo '--- requirements-lib.lock ---' && pip freeze --exclude-editable | grep -v '^carma'"
	docker run --rm -v "$(CURDIR):/repo:ro" python:3.11-slim-bookworm bash -c "cd /tmp && cp -r /repo/pyproject.toml /repo/README.md /repo/src /tmp/ && pip install --quiet '/tmp[dev]' && echo '--- requirements-dev.lock ---' && pip freeze --exclude-editable | grep -v '^carma'"
