import polars as pl

from examples.data_loader import load_close_data
from positionbt import BacktestVisualizer, PositionBacktester

# Load close data
close_df = load_close_data()

# Generate position data
position_df = close_df.select(pl.col("time")).with_columns(pl.lit(1).alias("position"))

# Initialize backtester with parameters
backtester = PositionBacktester(
    close_df=close_df,
    commission_rate=0.001,  # 0.1% commission rate
    annual_trading_days=365,  # Use 365 trading days per year
    indicators="all",  # Calculate all available indicators
)

# Run backtest
backtest_result = backtester.run(position_df)

# Print backtest results in tabular format
backtest_result.print()

# Create visualizer and show results in browser
visualizer = BacktestVisualizer()
visualizer.show_in_browser(backtest_result, backtester.params)
