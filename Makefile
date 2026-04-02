.PHONY: run test docker-build docker-run logs clean help

## ── Dev ──────────────────────────────────────────────────────────────────────

run:        ## Start FastAPI server with hot reload
	uvicorn app.main:app --reload --port 8000

test:       ## Run full test suite
	python -m pytest test_env.py -v

test-q:     ## Run tests (quiet)
	python -m pytest test_env.py -q

lint:       ## Run ruff linter
	ruff check app/ inference.py

## ── Docker ───────────────────────────────────────────────────────────────────

docker-build:   ## Build Docker image
	docker build -t email-triage-openenv .

docker-run:     ## Run Docker container
	docker run --rm -p 8000:8000 email-triage-openenv

compose-up:     ## Start with docker compose
	docker compose up --build

compose-down:   ## Stop docker compose
	docker compose down

## ── Inference ────────────────────────────────────────────────────────────────

run-easy:   ## Run inference agent on easy task
	python inference.py --task easy_triage --verbose

run-medium: ## Run inference agent on medium task
	python inference.py --task medium_triage --verbose

run-hard:   ## Run inference agent on hard task
	python inference.py --task hard_triage --verbose

run-all:    ## Run inference agent on all tasks
	python inference.py --all --verbose

## ── Utility ──────────────────────────────────────────────────────────────────

logs:       ## Show recent logs
	ls -la logs/ 2>/dev/null || echo "No logs yet — run an episode first"

clean:      ## Remove cache and logs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	rm -rf .pytest_cache logs/

help:       ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'
