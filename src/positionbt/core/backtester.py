import numpy as np
import pandas as pd
import polars as pl
from numba import njit

from positionbt.indicators.base import BaseIndicator
from positionbt.indicators.registry import indicator_registry
from positionbt.models.models import BacktestResult
from positionbt.utils.validation import (
    ValidationError,
    validate_and_convert_input,
    validate_annual_trading_days,
    validate_commission_rate,
    validate_data_type,
    validate_indicators,
    validate_time_alignment,
)


class PositionBacktester:
    def __init__(
        self,
        close_df: pl.DataFrame | pd.DataFrame,
        commission_rate: float = 0.0,
        annual_trading_days: int = 252,
        indicators: str | list[str] = "all",
    ) -> None:
        try:
            # Validate parameters
            validate_commission_rate(commission_rate)
            validate_annual_trading_days(annual_trading_days)
            self.sorted_indicators = validate_indicators(indicators)

            # Validate and convert input data
            self.close_df = validate_and_convert_input(close_df, data_type="close")
            self.commission_rate = commission_rate
            self.annual_trading_days = annual_trading_days
            self.indicators = indicators

        except ValidationError as e:
            raise ValueError(f"Invalid input: {e!s}")

    def _calculate_equity_curve(self, df: pl.DataFrame) -> pl.DataFrame:
        @njit(cache=True)
        def calculate_equity(
            close_arr: np.ndarray,
            position_arr: np.ndarray,
            return_arr: np.ndarray,
            position_change_arr: np.ndarray,
            commission_rate: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            Calculate equity curve and commission costs for a trading strategy.

            Args:
                close_arr: Array of closing prices
                position_arr: Array of target positions
                return_arr: Array of price returns
                position_change_arr: Array indicating position changes
                commission_rate: Commission rate for trading

            Returns:
                Tuple of (equity_curve, commission_cost) arrays
            """
            n = len(close_arr)

            # Initialize arrays
            equity_curve = np.ones(n, dtype=np.float32)  # Starting with 1 as base equity
            position_value = np.zeros(n, dtype=np.float32)  # Current position value
            commission_cost = np.zeros(n, dtype=np.float32)  # Trading commission costs

            # Initialize first position
            current_position = position_arr[0]

            # Set initial position value and commission if there's a position
            if current_position != 0:
                position_value[0] = current_position * equity_curve[0]
                commission_cost[0] = abs(current_position) * commission_rate * equity_curve[0]

            # Main calculation loop
            for i in range(1, n):
                # Calculate previous period's net position value (after commission)
                prev_net_position = position_value[i - 1] - commission_cost[i - 1]

                # Update position value based on returns
                position_value[i] = prev_net_position * (1 + return_arr[i])

                # Update equity curve
                equity_curve[i] = equity_curve[i - 1] + (position_value[i] - prev_net_position)

                # Handle position changes and commission
                if position_change_arr[i] != 0:
                    # Calculate commission for position change
                    commission_cost[i] = abs(position_arr[i] - current_position) * commission_rate * equity_curve[i]
                    current_position = position_arr[i]  # Update to new target position
                else:
                    # Update current position based on position value and equity
                    current_position = position_value[i] / equity_curve[i]

                # Handle account liquidation
                if equity_curve[i] <= 0:
                    equity_curve[i:] = 0  # Set remaining equity to 0
                    break

            return equity_curve, commission_cost

        close_arr = df["close"].to_numpy().astype(np.float32)
        position_arr = df["position"].to_numpy().astype(np.float32)
        return_arr = df["close"].pct_change().to_numpy().astype(np.float32)
        position_change_arr = df["position"].diff().abs().fill_null(pl.col("position").first()).to_numpy().astype(np.float32)

        equity_curve, commission_cost = calculate_equity(close_arr, position_arr, return_arr, position_change_arr, self.commission_rate)

        return df.select("time").with_columns(
            equity_curve=pl.lit(equity_curve),
            commission_cost=pl.lit(commission_cost),
            net_returns=pl.lit(equity_curve).pct_change().fill_null(0),
        )

    def add_indicator(self, indicator: BaseIndicator) -> None:
        """Add custom indicator

        Args:
            indicator: Custom indicator instance inheriting from BaseIndicator

        """
        indicator_registry.register(indicator)

    def run(self, position_df: pl.DataFrame | pd.DataFrame, **kwargs) -> BacktestResult:
        """Run backtest and return results

        Args:
            position_df: DataFrame containing position data
            **kwargs: Additional keyword arguments to be passed to indicators via cache

        Returns:
            BacktestResult: Object containing backtest results

        Raises:
            ValueError: If position data is invalid

        """
        try:
            # Validate and convert position data
            position_df = validate_and_convert_input(position_df, data_type="position").fill_null(0.0)

            # Validate time alignment
            validate_time_alignment(self.close_df, position_df)

            # Merge data and sort by time
            merged_df = self.close_df.join(
                position_df,
                on="time",
                how="inner",
            ).sort("time")

        except ValidationError as e:
            raise ValueError(f"Invalid position input: {e!s}")

        # Calculate equity curve
        merged_df = merged_df.join(self._calculate_equity_curve(merged_df), on="time")

        # Prepare cache
        cache = self._prepare_cache(merged_df)
        cache.update(kwargs)

        # Calculate indicators
        results = self._calculate_indicators(cache)

        # Split results into two dictionaries
        dataframes = {k: v[k] for k, v in results["dataframes"].items()}
        indicators = {k: v["value"] for k, v in results["indicators"].items()}
        formatted_indicators = {k: v["formatted_value"] for k, v in results["indicators"].items()}

        # Create and return BacktestResult object
        return BacktestResult(
            _dataframes={
                "merged_df": cache["merged_df"],
                "trade_records": cache["trade_records"],
                **dataframes,
            },
            _indicator_values=indicators,
            _formatted_indicator_values=formatted_indicators,
        )

    def _prepare_cache(self, merged_df: pl.DataFrame) -> dict:
        """Prepare calculation cache

        Args:
            merged_df: DataFrame containing time, equity_curve and returns columns

        Returns:
            Dict containing cached data needed for calculations

        """
        times = merged_df.get_column("time")

        # Process position and trade sequence
        merged_df = (
            merged_df.with_columns(
                pl.col("position").forward_fill().fill_null(0.0),
                prev_position=pl.col("position").shift(1).fill_null(0),
            )
            .with_columns(position_changed=(pl.col("position") != pl.col("prev_position")) & (pl.col("position") != 0))
            .with_columns(trade_seq=pl.when(pl.col("position") != 0).then(pl.col("position_changed").cum_sum()).otherwise(None))
            .drop("prev_position", "position_changed")
        )

        # Generate trade records
        trade_records = (
            # Filter valid trades and add next period data
            merged_df.filter(pl.col("trade_seq").is_not_null())
            .with_columns(
                pl.col("time").shift(-1).alias("next_time"),
                pl.col("equity_curve").shift(-1).fill_null(pl.col("equity_curve").last()).alias("next_equity_curve"),
                pl.col("close").shift(1).fill_null(pl.col("close").first()).alias("prev_close"),
            )
            # Group by trade sequence and aggregate
            .group_by("trade_seq")
            .agg(
                pl.col("position").first().alias("position"),
                pl.col("time").min().alias("entry_time"),
                pl.col("next_time").max().alias("exit_time"),
                pl.col("prev_close").first().alias("entry_price"),
                pl.col("close").last().alias("exit_price"),
                pl.col("equity_curve").first().alias("entry_equity"),
                pl.col("next_equity_curve").last().alias("exit_equity"),
            )
            # Calculate returns
            .with_columns(
                (pl.col("exit_equity") / pl.col("entry_equity") - 1).alias("return_rate"),
                (pl.col("exit_price") / pl.col("entry_price") - 1).alias("price_return"),
            )
            .sort("trade_seq")
        )

        # Calculate total days
        total_days = max((times[-1] - times[0]).days, 1)

        # Calculate data frequency (in days)
        time_diffs = times.diff().drop_nulls()
        time_diffs = time_diffs.filter(time_diffs > pd.Timedelta(minutes=0))
        avg_interval = float(time_diffs.min().total_seconds()) / (24 * 3600)

        # Calculate periods per day
        periods_per_day = 1 / avg_interval

        return {
            "merged_df": merged_df,
            "trade_records": trade_records,
            "annual_trading_days": self.annual_trading_days,
            "total_days": total_days,
            "periods_per_day": periods_per_day,
        }

    def _calculate_indicators(self, cache: dict) -> dict:
        """Calculate indicators

        Args:
            cache: Calculation cache dictionary

        Returns:
            Dict containing calculation results with both DataFrames and indicator values

        """
        result = dict()
        result["dataframes"] = dict()
        result["indicators"] = dict()
        for indicator in self.sorted_indicators:
            value = validate_data_type(indicator_registry.get_indicator(indicator).calculate(cache))

            if self.indicators != "all" and indicator not in self.indicators:
                continue

            if isinstance(value, pl.DataFrame):
                result["dataframes"][indicator] = value
            else:
                formatted_value = indicator_registry.get_indicator(indicator).format(value)
                result["indicators"][indicator] = {
                    "value": value,
                    "formatted_value": formatted_value,
                }

        return result

    @property
    def params(self) -> dict:
        """Get backtester parameters

        Returns:
            Dict containing backtester parameters:
                - commission_rate: Commission rate
                - annual_trading_days: Number of tradang q as per year
                - indicators: List of indicators to calculate

        """
        return {
            "commission_rate": self.commission_rate,
            "annual_trading_days": self.annual_trading_days,
            "indicators": list(self.sorted_indicators),
        }
