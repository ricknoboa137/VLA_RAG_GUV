.PHONY: check lint format types test pins smoke

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
