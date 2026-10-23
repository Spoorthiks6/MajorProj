PYTHON ?= python3
VENV_PYTHON := .venv/bin/python

.DEFAULT_GOAL := help
.PHONY: help setup install init-config pipeline dashboard

help:
	@echo "Available targets:"
	@echo "  make setup       Create the virtual environment and local config templates"
	@echo "  make install     Install requirements into .venv"
	@echo "  make init-config Copy example config files without overwriting local files"
	@echo "  make pipeline    Run the data and report pipeline"
	@echo "  make dashboard   Start the live dashboard and chat server"

setup: install init-config
	@echo "Setup complete. Edit config/activity_map.json and add your GROQ_API_KEY to .env."

install:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install -r requirements.txt

init-config:
	@test -f config/activity_map.json || cp config/activity_map.example.json config/activity_map.json
	@test -f .env || cp .env.example .env
	@echo "Created missing local config templates; existing files were left unchanged."

pipeline:
	$(PYTHON) run_pipeline.py

dashboard:
	$(PYTHON) src/09_dashboard_server.py
