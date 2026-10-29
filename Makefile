.PHONY: test lint smoke planner-smoke reproduce resume report verify verify-exact paper

SMOKE_ARGS = --steps 2 --seeds 11 --eval-size 8 --latency-repeats 3 --checkpoints --output runs/smoke

test:
	python -m pytest -q

lint:
	python -m ruff check .
	python -m ruff format --check .

smoke:
	python train_and_eval.py $(SMOKE_ARGS) --overwrite
	python train_and_eval.py $(SMOKE_ARGS) --resume
	python scripts/verify_results.py --results runs/smoke --output runs/smoke/verification.json

planner-smoke:
	python scripts/planner_demo.py --steps 4 --output runs/planner-smoke --overwrite

reproduce:
	python train_and_eval.py --steps 300 --seeds 11 22 33 --checkpoints --output runs/reference

resume:
	python train_and_eval.py --steps 300 --seeds 11 22 33 --checkpoints --resume --output runs/reference

report:
	python scripts/summarize.py results/metrics.json
	python scripts/audit_intents.py

verify:
	python scripts/verify_results.py --allow-source-changes --output runs/checkpoint-verification.json

verify-exact:
	python scripts/verify_results.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
