.PHONY: install dev test lint fmt eval eval-compare up down logs psql

install:        ## install deps (incl. dev) into .venv
	uv sync

dev:            ## run API with auto-reload on http://localhost:8000/docs
	uv run uvicorn app.main:app --reload

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

fmt:
	uv run ruff format . && uv run ruff check . --fix

eval:           ## eval current prompt version against the CI thresholds
	uv run python -m evals.run_eval --check evals/thresholds.json

eval-compare:   ## compare prompt versions on the golden set
	uv run python -m evals.run_eval --versions v1 v2

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api

psql:
	docker compose exec db psql -U adlens -d adlens
