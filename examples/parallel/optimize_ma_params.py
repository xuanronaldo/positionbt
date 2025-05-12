import multiprocessing as mp

import polars as pl

from examples.data_loader import load_close_data
from positionbt import PositionBacktester


def prepare_ma_data(close_df: pl.DataFrame, min_window: int, max_window: int, step: int) -> pl.DataFrame:
    """Pre-calculate all moving average data

    Args:
        close_df: DataFrame containing close prices
        min_window: Minimum MA window size
        max_window: Maximum MA window size
        step: Step size between window sizes

    Returns:
        DataFrame with all calculated moving averages
    """
    return close_df.with_columns(
        [
            pl.col("close").rolling_mean(window).alias(f"ma_{window}")
            for window in range(min_window, max_window + 1, step)
        ]
    )


def run_ma_strategy(
    ma_data: pl.DataFrame, fast_window: int, slow_window: int, backtester: PositionBacktester
) -> dict:
    """Run a single moving average strategy and return results

    Args:
        ma_data: DataFrame with pre-calculated moving averages
        fast_window: Fast MA window size
        slow_window: Slow MA window size
        backtester: Backtester instance

    Returns:
        Dictionary containing backtest results
    """
    # Generate position signals based on MA crossover
    position_df = ma_data.select(
        pl.col("time"),
        pl.col(f"ma_{fast_window}").alias("fast_line"),
        pl.col(f"ma_{slow_window}").alias("slow_line"),
    ).with_columns(pl.when(pl.col("fast_line") >= pl.col("slow_line")).then(1).otherwise(0).alias("position"))

    # Run backtest and add strategy identifiers
    results = backtester.run(position_df).indicator_values
    results["id"] = f"{fast_window}_{slow_window}"
    return results


def main():
    # Strategy parameters
    MIN_WINDOW = 10
    MAX_WINDOW = 500
    STEP = 10
    COMMISSION = 0.001
    TRADING_DAYS = 365

    # Load price data
    close_df = load_close_data()

    # Initialize backtester with settings
    backtester = PositionBacktester(
        close_df=close_df,
        commission_rate=COMMISSION,
        annual_trading_days=TRADING_DAYS,
        indicators=["sharpe_ratio"],
    )

    # Pre-calculate all required moving averages
    ma_data = prepare_ma_data(close_df, MIN_WINDOW, MAX_WINDOW, STEP)

    # Generate all parameter combinations for testing
    params = [
        (ma_data, fast, slow, backtester)
        for fast in range(MIN_WINDOW, MAX_WINDOW + 1, STEP)
        for slow in range(fast + STEP, MAX_WINDOW + 1, STEP)
    ]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(run_ma_strategy, params)

    # Convert results to DataFrame and sort by Sharpe ratio
    [fast_window, slow_window] = (
        pl.DataFrame(results).sort("sharpe_ratio", descending=True).item(0, "id").split("_")
    )

    print(f"Fast window: {fast_window}, Slow window: {slow_window}")


if __name__ == "__main__":
    main()
