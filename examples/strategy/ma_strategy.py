import polars as pl

from examples.data_loader import load_close_data
from positionbt import BacktestVisualizer, PositionBacktester

close_df = load_close_data()

backtester = PositionBacktester(close_df, commission_rate=0.001, annual_trading_days=365)


def run_ma_strategy(fast_window: int, slow_window: int) -> dict:
    position_df = close_df.with_columns(
        [
            (pl.col("close").rolling_mean(fast_window) >= pl.col("close").rolling_mean(slow_window))
            .cast(pl.Int8)
            .alias("position")
        ]
    ).select(["time", "position"])

    backtest_result = backtester.run(position_df)

    backtest_result.print()

    visualizer = BacktestVisualizer()
    visualizer.show_in_browser(backtest_result, backtester.params, notes="MA Strategy")


run_ma_strategy(10, 310)
