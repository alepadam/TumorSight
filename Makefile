.PHONY: setup lint format test dashboard clean

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pre-commit install

lint:
	ruff check .
	black --check .

format:
	black .
	ruff check --fix .

test:
	pytest --cov=src --cov-report=term-missing

dashboard:
	streamlit run dashboard/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov
