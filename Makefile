PYTHON ?= python
PYTHONPATH := src

mcpt-scorecard:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/analysis/build_mcpt_scorecard.py reports/stage_02_priors/returns --runs 200 --block-size 12

execution-sim:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_execution_simulation.py $(ORDERS) --symbol $(SYMBOL)
