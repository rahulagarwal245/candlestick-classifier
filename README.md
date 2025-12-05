# Candlestick Classifier (Candles → Bullish/Bearish)

Repository for an image-based candlestick classifier trained on 30-candle windows.
- Ticker used: RELIANCE.NS
- Approach: generate candlestick PNGs (mplfinance) → train CNN → deploy via Streamlit

## Contents
- notebooks/stock_prediction.ipynb  # cleaned Colab notebook
- generate_candles.py              # script to produce dataset from yfinance
- streamlit_app/app.py             # streamlit application for inference
- models/best_candle_cnn.h5        # optionally host externally if >50MB
- requirements.txt
- .gitignore

## Quickstart (Colab)
1. Run the notebook `notebooks/stock_prediction.ipynb`.
2. Generate candles and train model.
3. Export `best_candle_cnn.h5` to `models/` or host externally.
4. Deploy Streamlit app (see deployment instructions below).

## License
MIT
