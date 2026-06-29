from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class PairConfig:
    pair: str
    filename: str


PAIR_CONFIGS = [
    PairConfig("EURUSD", "eurusdohlcba.xlsx"),
    PairConfig("GBPUSD", "gbpusdohlcba.xlsx"),
    PairConfig("USDCAD", "usdcadphlcba.xlsx"),
    PairConfig("USDJPY", "usdjpyohlcba.xlsx"),
]

START_DATE = "2016-01-01"
END_DATE = "2026-01-01"
DATA_COLUMNS = ["Date", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST"]
OUTPUT_STEM = "adf_bollinger_meanrev"

STARTING_CAPITAL = 100_000.0
NOTIONAL_PER_TRADE = 100_000.0
TRADING_DAYS_PER_YEAR = 252

BB_WINDOW = 20
BB_STD_MULTIPLIER = 2.0
ATR_WINDOW = 14
ATR_STOP_MULTIPLIER = 1.0

ADF_WINDOW = 90
ADF_P_THRESHOLD = 0.10
ADF_LATCH_DAYS = 5


def load_bfix_ohlc(filepath: Path) -> pd.DataFrame:
    # Match the BFIX workbook layout already used elsewhere in this folder.
    df = pd.read_excel(filepath, skiprows=6)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.loc[:, DATA_COLUMNS].copy()

    for column in DATA_COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["PX_LAST"])
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Allow the reusable `run_backtest(df)` entry point to accept standard OHLCV naming.
    rename_map = {
        "Open": "PX_OPEN",
        "High": "PX_HIGH",
        "Low": "PX_LOW",
        "Close": "PX_LAST",
    }
    normalized = df.copy()
    normalized = normalized.rename(columns=rename_map)

    required = {"PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST"}
    missing = required.difference(normalized.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")

    if "Date" in normalized.columns:
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
        normalized = normalized.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    else:
        normalized = normalized.reset_index(drop=True)
        normalized["Date"] = normalized.index

    return normalized


def rolling_adf_pvalues(series: pd.Series, window: int) -> pd.Series:
    # ADF is inherently iterative here because each window runs a full statistical test.
    p_values = np.full(len(series), np.nan, dtype="float64")

    for end_idx in range(window, len(series) + 1):
        sample = series.iloc[end_idx - window:end_idx]
        if sample.nunique() <= 1:
            continue
        try:
            p_values[end_idx - 1] = adfuller(sample, regression="c", autolag="AIC")[1]
        except (ValueError, np.linalg.LinAlgError):
            p_values[end_idx - 1] = np.nan

    return pd.Series(p_values, index=series.index, name="ADF_PValue")


def build_regime_latch(p_values: pd.Series, threshold: float, latch_days: int) -> pd.Series:
    # Once stationarity is detected, keep the regime "on" for the trigger day plus
    # the next `latch_days - 1` trading days. A fresh trigger resets the timer.
    active = np.zeros(len(p_values), dtype=bool)
    days_remaining = 0

    for idx, p_value in enumerate(p_values):
        if pd.notna(p_value) and p_value <= threshold:
            days_remaining = latch_days

        if days_remaining > 0:
            active[idx] = True
            days_remaining -= 1

    return pd.Series(active, index=p_values.index, name="Regime_Active")


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    # Use the classic true-range definition so gaps are included in volatility.
    prev_close = df["PX_LAST"].shift(1)
    tr = pd.concat(
        [
            df["PX_HIGH"] - df["PX_LOW"],
            (df["PX_HIGH"] - prev_close).abs(),
            (df["PX_LOW"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # Most indicator work is vectorized; only the rolling ADF remains loop-based.
    data = normalize_ohlcv_columns(df)
    close = data["PX_LAST"]

    data["BB_Mid"] = close.rolling(window=BB_WINDOW, min_periods=BB_WINDOW).mean()
    rolling_std = close.rolling(window=BB_WINDOW, min_periods=BB_WINDOW).std()
    data["BB_Upper"] = data["BB_Mid"] + BB_STD_MULTIPLIER * rolling_std
    data["BB_Lower"] = data["BB_Mid"] - BB_STD_MULTIPLIER * rolling_std
    data["ATR"] = compute_atr(data, ATR_WINDOW)
    data["ADF_PValue"] = rolling_adf_pvalues(close, ADF_WINDOW)
    data["Regime_Active"] = build_regime_latch(
        data["ADF_PValue"],
        threshold=ADF_P_THRESHOLD,
        latch_days=ADF_LATCH_DAYS,
    )

    prev_close = close.shift(1)
    prev_upper = data["BB_Upper"].shift(1)
    prev_lower = data["BB_Lower"].shift(1)
    prev_mid = data["BB_Mid"].shift(1)

    # Entry requires an actual band cross, not simply trading outside the band.
    data["Long_Entry_Signal"] = (
        data["Regime_Active"]
        & prev_close.ge(prev_lower)
        & close.lt(data["BB_Lower"])
    )
    data["Short_Entry_Signal"] = (
        data["Regime_Active"]
        & prev_close.le(prev_upper)
        & close.gt(data["BB_Upper"])
    )
    data["Long_Target_Signal"] = prev_close.le(prev_mid) & close.gt(data["BB_Mid"])
    data["Short_Target_Signal"] = prev_close.ge(prev_mid) & close.lt(data["BB_Mid"])
    return data


def simulate_strategy(
    df: pd.DataFrame,
    pair: str,
    starting_capital: float = STARTING_CAPITAL,
    notional_per_trade: float = NOTIONAL_PER_TRADE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Trade simulation is stateful because exits depend on the entry ATR captured
    # at the time the position was opened.
    data = df.copy()
    data["Position"] = 0
    data["Units"] = 0.0
    data["Entry_Price"] = np.nan
    data["Entry_ATR"] = np.nan
    data["Stop_Loss"] = np.nan
    data["Daily_PnL"] = 0.0
    data["Trade_PnL"] = 0.0
    data["Exit_Reason"] = ""

    trades: list[dict[str, object]] = []

    position = 0
    units = 0.0
    entry_price = np.nan
    entry_atr = np.nan
    stop_loss = np.nan
    entry_date = None
    previous_close = np.nan

    for idx, row in data.iterrows():
        close_price = float(row["PX_LAST"])
        high_price = float(row["PX_HIGH"])
        low_price = float(row["PX_LOW"])
        day_pnl = 0.0
        trade_pnl = 0.0
        exit_reason = ""

        if position != 0 and pd.notna(previous_close):
            # Mark-to-market daily PnL lets us build an equity curve and drawdown series.
            day_pnl = position * units * (close_price - previous_close)

        if position == 1:
            hit_stop = low_price <= stop_loss
            hit_target = bool(row["Long_Target_Signal"])

            if hit_stop or hit_target:
                # Stops are executed at the precomputed stop level; profit exits use the
                # close of the bar that crossed the Bollinger midline.
                exit_price = stop_loss if hit_stop else close_price
                exit_reason = "Stop Loss" if hit_stop else "Take Profit"
                if pd.notna(previous_close):
                    day_pnl = position * units * (exit_price - previous_close)
                trade_pnl = position * units * (exit_price - entry_price)
                trades.append(
                    {
                        "Pair": pair,
                        "Side": "Long",
                        "Entry Date": entry_date,
                        "Exit Date": row["Date"],
                        "Entry Price": round(entry_price, 6),
                        "Exit Price": round(exit_price, 6),
                        "Entry ATR": round(entry_atr, 6),
                        "Stop Loss": round(stop_loss, 6),
                        "Units": round(units, 2),
                        "PnL $": round(trade_pnl, 2),
                        "Return %": round(100 * trade_pnl / starting_capital, 4),
                        "Exit Reason": exit_reason,
                    }
                )
                position = 0
                units = 0.0
                entry_price = np.nan
                entry_atr = np.nan
                stop_loss = np.nan
                entry_date = None

        elif position == -1:
            hit_stop = high_price >= stop_loss
            hit_target = bool(row["Short_Target_Signal"])

            if hit_stop or hit_target:
                exit_price = stop_loss if hit_stop else close_price
                exit_reason = "Stop Loss" if hit_stop else "Take Profit"
                if pd.notna(previous_close):
                    day_pnl = position * units * (exit_price - previous_close)
                trade_pnl = position * units * (exit_price - entry_price)
                trades.append(
                    {
                        "Pair": pair,
                        "Side": "Short",
                        "Entry Date": entry_date,
                        "Exit Date": row["Date"],
                        "Entry Price": round(entry_price, 6),
                        "Exit Price": round(exit_price, 6),
                        "Entry ATR": round(entry_atr, 6),
                        "Stop Loss": round(stop_loss, 6),
                        "Units": round(units, 2),
                        "PnL $": round(trade_pnl, 2),
                        "Return %": round(100 * trade_pnl / starting_capital, 4),
                        "Exit Reason": exit_reason,
                    }
                )
                position = 0
                units = 0.0
                entry_price = np.nan
                entry_atr = np.nan
                stop_loss = np.nan
                entry_date = None

        if position == 0 and pd.notna(row["ATR"]):
            # Store the ATR observed on the entry bar so the stop never drifts later.
            if bool(row["Long_Entry_Signal"]):
                position = 1
                entry_price = close_price
                entry_atr = float(row["ATR"])
                stop_loss = entry_price - ATR_STOP_MULTIPLIER * entry_atr
                units = notional_per_trade / entry_price
                entry_date = row["Date"]
            elif bool(row["Short_Entry_Signal"]):
                position = -1
                entry_price = close_price
                entry_atr = float(row["ATR"])
                stop_loss = entry_price + ATR_STOP_MULTIPLIER * entry_atr
                units = notional_per_trade / entry_price
                entry_date = row["Date"]

        data.at[idx, "Position"] = position
        data.at[idx, "Units"] = units
        data.at[idx, "Entry_Price"] = entry_price
        data.at[idx, "Entry_ATR"] = entry_atr
        data.at[idx, "Stop_Loss"] = stop_loss
        data.at[idx, "Daily_PnL"] = day_pnl
        data.at[idx, "Trade_PnL"] = trade_pnl
        data.at[idx, "Exit_Reason"] = exit_reason

        previous_close = close_price

    trade_log = pd.DataFrame(trades)
    data["Strategy_Equity"] = starting_capital + data["Daily_PnL"].cumsum()
    buy_hold_units = starting_capital / data["PX_LAST"].iloc[0]
    data["BuyHold_Equity"] = buy_hold_units * data["PX_LAST"]
    data["Strategy_Return"] = data["Strategy_Equity"].pct_change().fillna(0.0)
    return data, trade_log


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve.div(running_max).sub(1.0)
    return float(drawdown.min())


def summarize_backtest(
    result_df: pd.DataFrame,
    trade_log: pd.DataFrame,
    pair: str,
    starting_capital: float = STARTING_CAPITAL,
) -> pd.DataFrame:
    # Keep the summary compact and aligned with the metrics the user requested.
    final_equity = float(result_df["Strategy_Equity"].iloc[-1])
    total_return = 100 * (final_equity / starting_capital - 1.0)
    max_dd = 100 * max_drawdown(result_df["Strategy_Equity"])

    if trade_log.empty:
        win_rate = np.nan
        completed_trades = 0
        avg_trade_pnl = np.nan
    else:
        completed_trades = len(trade_log)
        win_rate = round(100 * (trade_log["PnL $"] > 0).mean(), 2)
        avg_trade_pnl = round(float(trade_log["PnL $"].mean()), 2)

    ann_vol = result_df["Strategy_Return"].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (
        result_df["Strategy_Return"].mean() / result_df["Strategy_Return"].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if result_df["Strategy_Return"].std() > 0
        else np.nan
    )

    summary = pd.DataFrame(
        [
            {
                "Pair": pair,
                "Completed Trades": completed_trades,
                "Win Rate %": win_rate,
                "Average Trade PnL $": avg_trade_pnl,
                "Regime Active %": round(100 * result_df["Regime_Active"].mean(), 2),
                "Final Strategy Equity $": round(final_equity, 2),
                "Final BuyHold Equity $": round(float(result_df["BuyHold_Equity"].iloc[-1]), 2),
                "Total Return %": round(total_return, 2),
                "BuyHold Return %": round(
                    100 * (result_df["BuyHold_Equity"].iloc[-1] / starting_capital - 1.0), 2
                ),
                "Max Drawdown %": round(max_dd, 2),
                "BuyHold Max Drawdown %": round(100 * max_drawdown(result_df["BuyHold_Equity"]), 2),
                "Annualized Vol %": round(100 * ann_vol, 2),
                "Sharpe": round(float(sharpe), 3) if pd.notna(sharpe) else np.nan,
            }
        ]
    )
    return summary


def plot_equity_curve(result_df: pd.DataFrame, pair: str, output_path: Path) -> None:
    # Keep the chart consistent with the existing backtest workflow: strategy
    # equity against a simple buy-and-hold benchmark.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result_df["Date"], result_df["Strategy_Equity"], label="Strategy Equity", linewidth=1.8)
    ax.plot(result_df["Date"], result_df["BuyHold_Equity"], label="Buy and Hold", linewidth=1.4, alpha=0.85)
    ax.set_title(f"{pair} ADF Bollinger Mean-Reversion Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_backtest_for_pair(config: PairConfig, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_dir = Path(__file__).resolve().parent
    raw_df = load_bfix_ohlc(base_dir / config.filename)
    feature_df = prepare_features(raw_df)
    result_df, trade_log = simulate_strategy(feature_df, pair=config.pair)
    summary_df = summarize_backtest(result_df, trade_log, pair=config.pair)

    prefix = config.pair.lower()
    result_df.to_csv(output_dir / f"{prefix}_{OUTPUT_STEM}_signals.csv", index=False)
    trade_log.to_csv(output_dir / f"{prefix}_{OUTPUT_STEM}_trades.csv", index=False)
    summary_df.to_csv(output_dir / f"{prefix}_{OUTPUT_STEM}_summary.csv", index=False)
    plot_equity_curve(result_df, config.pair, output_dir / f"{prefix}_{OUTPUT_STEM}_equity_curve.png")
    return result_df, trade_log, summary_df


def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_df = prepare_features(df)
    result_df, trade_log = simulate_strategy(feature_df, pair="INPUT_DF")
    summary_df = summarize_backtest(result_df, trade_log, pair="INPUT_DF")
    return result_df, trade_log, summary_df


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "backtest_outputs"
    output_dir.mkdir(exist_ok=True)

    combined_summary: list[pd.DataFrame] = []

    for config in PAIR_CONFIGS:
        _, trade_log, summary_df = run_backtest_for_pair(config, output_dir)
        combined_summary.append(summary_df)

        print(f"\n=== {config.pair} ADF Bollinger Mean-Reversion Summary ===")
        print(summary_df.to_string(index=False))
        print("\nRecent trades:")
        if trade_log.empty:
            print("No completed trades.")
        else:
            print(trade_log.tail(10).to_string(index=False))

    combined_summary_df = pd.concat(combined_summary, ignore_index=True)
    combined_summary_df.to_csv(output_dir / f"combined_{OUTPUT_STEM}_summary.csv", index=False)
    print(f"\nDetailed outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
