# Convenience targets. Everything is also available as `psychad <subcommand>`.
PROFILE ?= dev
PSY = psychad --profile $(PROFILE)

.PHONY: install test smoke inspect build eda splits aim1 controls aim2 transfer importance de scaling report

install:
	pip install -e ".[all]"

test:
	pytest -q

smoke:           ## end-to-end run on synthetic data (no network, ~1-2 min)
	python scripts/smoke_test.py

inspect:         ## list cohorts / diagnosis strings / donor counts from CELLxGENE
	$(PSY) inspect

build:           ## extract + subsample + HVG -> data/derived/<profile>/
	$(PSY) build

eda:
	$(PSY) eda

splits:
	$(PSY) splits

aim1:
	$(PSY) benchmark

controls:
	$(PSY) controls

aim2:
	$(PSY) celltype

transfer:
	$(PSY) transfer

importance:
	$(PSY) importance

de:
	$(PSY) de

scaling:
	$(PSY) scaling

report:
	$(PSY) report
