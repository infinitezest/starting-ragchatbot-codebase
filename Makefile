.PHONY: format check test quality

## Format all Python files with black
format:
	uv run black .

## Check formatting without making changes (for CI)
check:
	uv run black --check .

## Run the test suite
test:
	cd backend && uv run pytest tests/ -v

## Run all quality checks (formatting + tests)
quality: check test
