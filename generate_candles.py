#!/usr/bin/env python3
"""
generate_candles.py
Script: download OHLCV for a ticker, label next-day movement, and save candlestick PNGs.
Usage: python generate_candles.py --ticker RELIANCE.NS --start 2015-01-01 --end 2024-12-31
"""

import argparse
import os
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from tqdm import tqdm

def main(ticker, start, end, out_base="candles_dataset", window=30, dpi=100, max_images=None):
    os.makedirs(out_base, exist_ok=True)
    bull_dir = os.path.join(out_base, "bullish")
    bear_dir = os.path.join(out_base, "bearish")
    os.makedirs(bull_dir, exist_ok=True)
    os.makedirs(bear_dir, exist_ok=True)

    print(f"Downloading {ticker} from {start} to {end} ...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise SystemExit("Empty DataFrame from yfinance")

    df = data.copy()
    # flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.dropna(subset=["Open","High","Low","Close","Volume"])
    df["Next_Close"] = df["Close"].shift(-1)
    df = df.dropna(subset=["Next_Close"])
    df["Label"] = (df["Next_Close"] > df["Close"]).astype(int)

    ohlc_df = df[["Open","High","Low","Close"]]

    total = 0
    start_idx = window
    for i in tqdm(range(start_idx, len(df)), desc="windows"):
        wstart = i - window
        we = i
        chunk = ohlc_df.iloc[wstart:we]
        if len(chunk) != window:
            continue
        label = int(df["Label"].iloc[i-1])
        out_dir = bull_dir if label == 1 else bear_dir
        fname = os.path.join(out_dir, f"candle_{i}.png")
        mpf.plot(chunk, type='candle', style='charles', volume=False, axisoff=True,
                 savefig=dict(fname=fname, dpi=dpi, bbox_inches="tight", pad_inches=0.05))
        total += 1
        if max_images and total >= max_images:
            break
    print(f"Saved {total} images to {out_base}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="RELIANCE.NS")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--out", default="candles_dataset")
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--max_images", type=int, default=None)
    args = p.parse_args()
    main(args.ticker, args.start, args.end, out_base=args.out, window=args.window, dpi=args.dpi, max_images=args.max_images)
