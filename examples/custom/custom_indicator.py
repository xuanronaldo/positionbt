import polars as pl

from examples.data_loader import load_close_data
from positionbt import BaseIndicator, PositionBacktester, indicator_registry


class MonthlyReturn(BaseIndicator):
    """Monthly return indicator"""

    @property
    def name(self) -> str:
        return "monthly_return"

    @property
    def requires(self) -> set[str]:
        # Depends on annual return
        return {"annual_return"}

    def calculate(self, cache: dict) -> float:
        """Calculate monthly return

        Calculation method:
        1. Convert from annual return
        2. Using formula: (1 + r_annual)^(1/12) - 1

        Args:
            cache: Dictionary containing calculation cache

        Returns:
            Monthly return value

        """
        if "monthly_return" not in cache:
            annual_return = cache["annual_return"]
            monthly_return = (1 + annual_return) ** (1 / 12) - 1
            cache["monthly_return"] = monthly_return

        return cache["monthly_return"]

    def format(self, value: float) -> str:
        """Format monthly return value as percentage

        Args:
            value: Monthly return value

        Returns:
            Formatted string with percentage

        """
        return f"{value:.2%}"


# Register custom indicator
indicator_registry.register(MonthlyReturn())

# Load close data
close_df = load_close_data()

# Generate position data
position_df = close_df.select(pl.col("time")).with_columns(pl.lit(1).alias("position"))

# Create backtester instance (using all indicators including the newly registered monthly return)
backtester = PositionBacktester(
    close_df=close_df,
    commission_rate=0.001,  # 0.1% commission rate
    annual_trading_days=365,  # Use 365 trading days per year
    indicators=["monthly_return"],  # Use all registered indicators
)

# Run backtest
results = backtester.run(position_df)

# Print results
results.print()
