.PHONY: dev test lint migrate clean type-check \
	gen-certs prod-build prod-up prod-down prod-clean \
	prod-ps prod-logs prod-logs-backend prod-logs-worker prod-logs-nginx prod-logs-frontend \
	prod-restart-backend prod-restart-worker smoke-test integration-test prod-pull prod-stats \
	eval eval-quick eval-report eval-clean compare-providers smoke-test-week16

dev:
	docker compose -f docker/docker-compose.yml up --build

test:
	cd backend && pytest tests/ -v

lint:
	cd backend && ruff check . && mypy .

type-check:
	cd frontend && npm run type-check

migrate:
	cd backend && alembic upgrade head

clean:
	docker compose -f docker/docker-compose.yml down -v

gen-certs:
	bash scripts/gen-certs.sh

prod-build:
	docker compose -f docker/docker-compose.prod.yml build

prod-up:
	@if [ ! -f .env.prod ]; then \
		echo "ERROR: .env.prod not found. Copy .env.prod.example and fill in values."; \
		exit 1; \
	fi
	@if [ ! -f docker/nginx/certs/localhost.crt ] || [ ! -f docker/nginx/certs/localhost.key ]; then \
		echo "TLS certs not found. Generating certs..."; \
		bash scripts/gen-certs.sh; \
	fi
	docker compose -f docker/docker-compose.prod.yml up -d

prod-down:
	docker compose -f docker/docker-compose.prod.yml down

prod-clean:
	docker compose -f docker/docker-compose.prod.yml down -v

prod-ps:
	docker compose -f docker/docker-compose.prod.yml ps

prod-logs:
	docker compose -f docker/docker-compose.prod.yml logs -f

prod-logs-backend:
	docker compose -f docker/docker-compose.prod.yml logs -f backend

prod-logs-worker:
	docker compose -f docker/docker-compose.prod.yml logs -f worker

prod-logs-nginx:
	docker compose -f docker/docker-compose.prod.yml logs -f nginx

prod-logs-frontend:
	docker compose -f docker/docker-compose.prod.yml logs -f frontend

prod-restart-backend:
	docker compose -f docker/docker-compose.prod.yml restart backend

prod-restart-worker:
	docker compose -f docker/docker-compose.prod.yml restart worker

smoke-test:
	bash scripts/smoke-test.sh

integration-test:
	cd backend && pytest tests/integration/ -v -m integration --tb=short

prod-pull:
	docker compose -f docker/docker-compose.prod.yml pull postgres redis qdrant neo4j nginx

prod-stats:
	docker stats \
	  codebase_postgres \
	  codebase_redis \
	  codebase_qdrant \
	  codebase_neo4j \
	  codebase_backend \
	  codebase_worker \
	  codebase_frontend \
	  codebase_nginx \
	  --no-stream

eval:
	cd backend && python evaluation/run_eval.py

eval-quick:
	cd backend && python evaluation/run_eval.py --repos psf/requests --questions 5 --skip-load-test

eval-report:
	cd backend && python evaluation/run_eval.py --report-only

eval-clean:
	rm -rf data/benchmarks/results data/benchmarks/latency data/benchmarks/costs data/benchmarks/quality data/benchmarks/load data/benchmarks/BENCHMARK_REPORT*.md data/benchmarks/BENCHMARK_REPORT*.html

compare-providers:
	cd backend && python evaluation/provider_comparison.py

smoke-test-week16:
	bash scripts/smoke-test-week16.sh
