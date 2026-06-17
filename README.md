# Möbius Transform of Correlation Matrix

Applies a Möbius transformation (fractional linear) to the eigenvalues of the ETF correlation matrix. The transformation parameter is a function of the composite macro factor (from all macro variables). The per‑ETF score is the sensitivity of the transformed spectral norm to macro changes – a measure of macro‑driven correlation structure.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Correlation matrix eigen-decomposition
- Möbius transform: λ' = λ / (c·λ + 1) with c = 1 + macro_factor
- Sensitivity via finite difference
- Score = derivative of spectral norm w.r.t. macro factor
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-mobius-transform-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High sensitivity → ETF correlation structure is highly responsive to macro changes.
- Low sensitivity → correlation structure is stable.

## Requirements

See `requirements.txt`.
