PY ?= python
CFG ?=

.PHONY: install check fetch prepare label train rq1 rq2 rq3 rq4 rq5 all smoke test clean-runs

install:
	$(PY) -m pip install -U pip "setuptools<81" wheel
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

check:
	$(PY) -m resa.cli check $(CFG)

fetch:
	$(PY) -m resa.cli fetch all $(CFG)

prepare:
	$(PY) -m resa.cli prepare $(CFG)

label:
	$(PY) -m resa.cli label --stage verify $(CFG)
	$(PY) -m resa.cli label --stage recover $(CFG)

rq1:
	$(PY) -m resa.cli rq1 $(CFG)
rq2:
	$(PY) -m resa.cli rq2 $(CFG)
rq3:
	$(PY) -m resa.cli rq3 $(CFG)
rq4:
	$(PY) -m resa.cli rq4 $(CFG)
rq5:
	$(PY) -m resa.cli rq5 $(CFG)

all:
	bash scripts/run_all.sh

smoke:
	bash scripts/smoke_test.sh

test:
	$(PY) -m pytest -q

# delete cached pipeline runs (keeps trained models and labels) after changing inference settings
clean-runs:
	rm -rf runs
