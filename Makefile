PYTHON ?= .venv/bin/python

.PHONY: setup index app phoenix cli graph eval eval-fast test
setup:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

index:
	$(PYTHON) -m restaurant_ai.rag.build_index

app:
	$(PYTHON) -m restaurant_ai.app.gradio_app

phoenix:
	.venv/bin/phoenix serve

cli:
	$(PYTHON) -m restaurant_ai.data_pipeline.manage_cli

graph:
	$(PYTHON) -m restaurant_ai.agents.graph --mermaid

eval:
	$(PYTHON) -m evals.run_evals

eval-fast:
	$(PYTHON) -m evals.run_evals --skip-agents

test:
	$(PYTHON) -m pytest -q
