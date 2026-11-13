# loan-approval

AI Lab coursework: Predicts whether a loan application is approved with a Decision Tree and shows the rules it learned, with an interactive Streamlit demo.

## Layout

- `loan_approval/` this project's code, starting with `spec.py` (the SPEC dictionaries)
- `pxlab/` vendored copy of the shared engine (see `pxlab/VERSION`). Don't edit it here; it is replaced on the next sync.
- `main.py` runs the full study and rewrites `docs/results.md`; `app.py` is the Streamlit demo; `app_overview.md` the demo's Overview text
- `outputs/` figures and CSV tables, `models/` saved models (both committed for the hosted app)
- `docs/` methodology, results, deploy

## Commands

- Full study: `uv run python main.py`
- Demo: `uv run streamlit run app.py`
- Lint: `uv run ruff check . && uv run ruff format --check .`

## Conventions

- Beginner-student style on purpose: plain functions, docstrings on every function, comments explaining the AI concepts.
- `random_state=42` everywhere. Preprocessing is fitted inside the pipeline on training data only.
- After changing the study, re-run `main.py` and refresh the README results and `app_overview.md`.
