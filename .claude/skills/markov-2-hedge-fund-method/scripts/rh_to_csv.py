#!/usr/bin/env python3
"""Convert a Robinhood MCP get_equity_historicals JSON dump to a markov2 CSV.

The MCP result has schema {data: {results: [{symbol, interval, bounds, bars: [
  {begins_at, open_price, high_price, low_price, close_price, volume, ...}]}]}}.

Usage:
    python rh_to_csv.py ROBINHOOD_RESULT.json OUT.csv [SYMBOL]
"""
import csv
import json
import sys


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    want = sys.argv[3].upper() if len(sys.argv) > 3 else None

    with open(src) as f:
        doc = json.load(f)
    results = doc.get("data", {}).get("results", [])
    if not results:
        print("no results in file")
        return 1
    res = next((r for r in results if r.get("symbol", "").upper() == want), results[0]) if want else results[0]
    bars = res.get("bars", [])

    rows = 0
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "volume"])
        for b in bars:
            if b.get("interpolated"):  # gap-fill bars carry no new info
                continue
            w.writerow([
                b["begins_at"][:10],
                b["open_price"], b["high_price"], b["low_price"],
                b["close_price"], b["volume"],
            ])
            rows += 1
    print(f"wrote {rows} rows for {res.get('symbol')} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
