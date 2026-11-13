# Deploy

The demo runs on Streamlit Community Cloud from this repo.

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app** > **Deploy a public app from GitHub**.
3. Repository `pxnz3r/loan-approval`, branch `main`, main file `app.py`.
4. **Advanced settings**: Python 3.14.
5. Deploy. Dependencies come from `pyproject.toml` and `uv.lock`.

The saved models and figures are committed, so the app doesn't train anything on start. After changing the study, run `uv run python main.py` and commit `models/` and `outputs/`.
