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
OUTPUT_STEM = "adf_price_action_meanrev"

STARTING_CAPITAL = 100_000.0
NOTIONAL_PER_TRADE = 100_000.0
TRADING_DAYS_PER_YEAR = 252

RANGE_WINDOW = 10
SMA_WINDOW = 10
ATR_WINDOW = 14
ATR_STOP_MULTIPLIER = 1.0
STREAK_LENGTH = 3
LOW_PERCENTILE_THRESHOLD = 0.10
HIGH_PERCENTILE_THRESHOLD = 0.90

ADF_WINDOW = 90
ADF_P_THRESHOLD = 0.10
ADF_LATCH_DAYS = 5


def load_bfix_ohlc(filepath: Path) -> pd.DataFrame:
    # Match the BFIX workbook layout already used in the existing FX scripts.
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


def normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Allow direct use with a standard OHLC DataFrame named Open/High/Low/Close.
    rename_map = {
        "Open": "PX_OPEN",
        "High": "PX_HIGH",
        "Low": "PX_LOW",
        "Close": "PX_LAST",
    }
    normalized = df.copy().rename(columns=rename_map)
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
    # Rolling ADF cannot be vectorized because each window runs a full hypothesis test.
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
    # A fresh stationarity trigger resets the latch to the full number of trading days.
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
    prev_close = df["PX_LAST"].shift(1)
    true_range = pd.concat(
        [
            df["PX_HIGH"] - df["PX_LOW"],
            (df["PX_HIGH"] - prev_close).abs(),
            (df["PX_LOW"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # Indicator and signal construction is vectorized except for the rolling ADF loop.
    data = normalize_ohlc_columns(df)
    close = data["PX_LAST"]
    close_change = close.diff()

    data["10_High"] = data["PX_HIGH"].rolling(window=RANGE_WINDOW, min_periods=RANGE_WINDOW).max()
    data["10_Low"] = data["PX_LOW"].rolling(window=RANGE_WINDOW, min_periods=RANGE_WINDOW).min()
    data["SMA_10"] = close.rolling(window=SMA_WINDOW, min_periods=SMA_WINDOW).mean()
    data["ATR"] = compute_atr(data, ATR_WINDOW)
    data["ADF_PValue"] = rolling_adf_pvalues(close, ADF_WINDOW)
    data["Regime_Active"] = build_regime_latch(data["ADF_PValue"], ADF_P_THRESHOLD, ADF_LATCH_DAYS)

    range_width = data["10_High"] - data["10_Low"]
    data["Percentile"] = np.where(
        range_width > 0,
        (close - data["10_Low"]) / range_width,
        np.nan,
    )

    down_day = close_change.lt(0)
    up_day = close_change.gt(0)
    data["Lower_Close_Streak"] = (
        down_day.rolling(window=STREAK_LENGTH, min_periods=STREAK_LENGTH).sum() == STREAK_LENGTH
    )
    data["Higher_Close_Streak"] = (
        up_day.rolling(window=STREAK_LENGTH, min_periods=STREAK_LENGTH).sum() == STREAK_LENGTH
    )

    prev_close = close.shift(1)
    prev_sma = data["SMA_10"].shift(1)
    data["Long_Entry_Signal"] = (
        data["Regime_Active"]
        & data["Lower_Close_Streak"]
        & data["Percentile"].lt(LOW_PERCENTILE_THRESHOLD)
    )
    data["Short_Entry_Signal"] = (
        data["Regime_Active"]
        & data["Higher_Close_Streak"]
        & data["Percentile"].gt(HIGH_PERCENTILE_THRESHOLD)
    )
    data["Long_Target_Signal"] = prev_close.le(prev_sma) & close.gt(data["SMA_10"])
    data["Short_Target_Signal"] = prev_close.ge(prev_sma) & close.lt(data["SMA_10"])
    return data


def simulate_strategy(
    df: pd.DataFrame,
    pair: str,
    starting_capital: float = STARTING_CAPITAL,
    notional_per_trade: float = NOTIONAL_PER_TRADE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Simulation is stateful because stops depend on the ATR captured on the entry bar.
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
    entry_idx = None
    previous_close = np.nan

    for idx, row in data.iterrows():
        close_price = float(row["PX_LAST"])
        high_price = float(row["PX_HIGH"])
        low_price = float(row["PX_LOW"])
        day_pnl = 0.0
        trade_pnl = 0.0
        exit_reason = ""

        if position != 0 and pd.notna(previous_close):
            day_pnl = position * units * (close_price - previous_close)

        if position == 1:
            hit_stop = low_price <= stop_loss
            hit_target = bool(row["Long_Target_Signal"])

            if hit_stop or hit_target:
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
                        "Holding Bars": idx - entry_idx,
                        "Holding Days": idx - entry_idx,
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
                entry_idx = None

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
                        "Holding Bars": idx - entry_idx,
                        "Holding Days": idx - entry_idx,
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
                entry_idx = None

        if position == 0 and pd.notna(row["ATR"]):
            if bool(row["Long_Entry_Signal"]):
                position = 1
                entry_price = close_price
                entry_atr = float(row["ATR"])
                stop_loss = entry_price - ATR_STOP_MULTIPLIER * entry_atr
                units = notional_per_trade / entry_price
                entry_date = row["Date"]
                entry_idx = idx
            elif bool(row["Short_Entry_Signal"]):
                position = -1
                entry_price = close_price
                entry_atr = float(row["ATR"])
                stop_loss = entry_price + ATR_STOP_MULTIPLIER * entry_atr
                units = notional_per_trade / entry_price
                entry_date = row["Date"]
                entry_idx = idx

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
    final_equity = float(result_df["Strategy_Equity"].iloc[-1])
    total_return = 100 * (final_equity / starting_capital - 1.0)
    max_dd = 100 * max_drawdown(result_df["Strategy_Equity"])

    if trade_log.empty:
        completed_trades = 0
        win_rate = np.nan
        winning_trades = 0
        losing_trades = 0
        avg_trade_pnl = np.nan
        average_win = np.nan
        average_loss = np.nan
        best_trade = np.nan
        worst_trade = np.nan
        profit_factor = np.nan
        average_holding_days = np.nan
    else:
        completed_trades = len(trade_log)
        win_rate = round(100 * (trade_log["PnL $"] > 0).mean(), 2)
        winning_trades = int((trade_log["PnL $"] > 0).sum())
        losing_trades = int((trade_log["PnL $"] <= 0).sum())
        avg_trade_pnl = round(float(trade_log["PnL $"].mean()), 2)
        average_win = round(float(trade_log.loc[trade_log["PnL $"] > 0, "PnL $"].mean()), 2) if winning_trades > 0 else np.nan
        average_loss = round(float(trade_log.loc[trade_log["PnL $"] <= 0, "PnL $"].mean()), 2) if losing_trades > 0 else np.nan
        best_trade = round(float(trade_log["PnL $"].max()), 2)
        worst_trade = round(float(trade_log["PnL $"].min()), 2)
        gross_profit = float(trade_log.loc[trade_log["PnL $"] > 0, "PnL $"].sum())
        gross_loss = float(-trade_log.loc[trade_log["PnL $"] <= 0, "PnL $"].sum())
        profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else np.nan
        average_holding_days = round(float(trade_log["Holding Days"].mean()), 2)

    strategy_returns = result_df["Strategy_Return"]
    ann_vol = strategy_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (
        strategy_returns.mean() / strategy_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if strategy_returns.std() > 0
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "Pair": pair,
                "Completed Trades": completed_trades,
                "Winning Trades": winning_trades,
                "Losing Trades": losing_trades,
                "Win Rate %": win_rate,
                "Average Trade PnL $": avg_trade_pnl,
                "Average Win $": average_win,
                "Average Loss $": average_loss,
                "Best Trade $": best_trade,
                "Worst Trade $": worst_trade,
                "Profit Factor": profit_factor,
                "Average Holding Days": average_holding_days,
                "Regime Active %": round(100 * result_df["Regime_Active"].mean(), 2),
                "Final Strategy Equity $": round(final_equity, 2),
                "Final BuyHold Equity $": round(float(result_df["BuyHold_Equity"].iloc[-1]), 2),
                "Total Return %": round(total_return, 2),
                "BuyHold Return %": round(
                    100 * (result_df["BuyHold_Equity"].iloc[-1] / starting_capital - 1.0),
                    2,
                ),
                "Max Drawdown %": round(max_dd, 2),
                "BuyHold Max Drawdown %": round(100 * max_drawdown(result_df["BuyHold_Equity"]), 2),
                "Annualized Vol %": round(100 * ann_vol, 2),
                "Sharpe": round(float(sharpe), 3) if pd.notna(sharpe) else np.nan,
            }
        ]
    )


def plot_equity_curve(result_df: pd.DataFrame, pair: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result_df["Date"], result_df["Strategy_Equity"], label="Strategy Equity", linewidth=1.8)
    ax.plot(result_df["Date"], result_df["BuyHold_Equity"], label="Buy and Hold", linewidth=1.4, alpha=0.85)
    ax.set_title(f"{pair} ADF Price-Action Mean-Reversion Equity Curve")
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
    output_dir = base_dir / "backtest_outputs_2"
    output_dir.mkdir(exist_ok=True)

    combined_summary: list[pd.DataFrame] = []

    for config in PAIR_CONFIGS:
        _, trade_log, summary_df = run_backtest_for_pair(config, output_dir)
        combined_summary.append(summary_df)

        print(f"\n=== {config.pair} ADF Price-Action Mean-Reversion Summary ===")
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
