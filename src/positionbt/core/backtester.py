from typing import Union

import pandas as pd
import polars as pl

from positionbt.indicators.base import BaseIndicator
from positionbt.indicators.registry import indicator_registry
from positionbt.models.models import BacktestResult
from positionbt.utils.validation import (
    ValidationError,
    validate_and_convert_input,
    validate_annual_trading_days,
    validate_commission,
    validate_data_type,
    validate_indicators,
    validate_time_alignment,
)


class PositionBacktester:
    def __init__(
        self,
        close_df: Union[pl.DataFrame, pd.DataFrame],
        commission: float = 0.0,
        annual_trading_days: int = 252,
        indicators: Union[str, list[str]] = "all",
    ) -> None:
        try:
            # Validate parameters
            validate_commission(commission)
            validate_annual_trading_days(annual_trading_days)
            self.sorted_indicators = validate_indicators(indicators)

            # Validate and convert input data
            self.close_df = validate_and_convert_input(close_df, data_type="close")
            self.commission = commission
            self.annual_trading_days = annual_trading_days
            self.indicators = indicators

        except ValidationError as e:
            raise ValueError(f"Invalid input: {e!s}")

    def _calculate_funding_curve(self, merged_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate funding curve

        Args:
            merged_df: DataFrame containing close prices and positions

        Returns:
            DataFrame with time, funding curve and returns

        """
        # Calculate returns
        returns = merged_df["close"].pct_change().fill_null(0)

        # Calculate position returns
        position_returns = returns * merged_df["position"].shift(1).fill_null(0)

        # Calculate transaction costs
        position_changes = merged_df["position"].diff().abs().fill_null(0)
        transaction_costs = position_changes * self.commission

        # Calculate net returns
        net_returns = position_returns - transaction_costs

        # Calculate funding curve
        funding_curve = (1 + net_returns).cum_prod()

        # Create result DataFrame
        result = pl.DataFrame(
            {
                "time": merged_df["time"],
                "funding_curve": funding_curve,
                "returns": net_returns,
            }
        )

        return result

    def add_indicator(self, indicator: BaseIndicator) -> None:
        """Add custom indicator

        Args:
            indicator: Custom indicator instance inheriting from BaseIndicator

        """
        indicator_registry.register(indicator)

    def run(self, position_df: Union[pl.DataFrame, pd.DataFrame]) -> BacktestResult:
        """Run backtest and return results

        Args:
            position_df: DataFrame containing position data

        Returns:
            BacktestResult: Object containing backtest results

        Raises:
            ValueError: If position data is invalid

        """
        try:
            # Validate and convert position data
            position_df = validate_and_convert_input(position_df, data_type="position")

            # Validate time alignment
            validate_time_alignment(self.close_df, position_df)

            # Merge data and sort by time
            merged_df = self.close_df.join(
                position_df.select(["time", "position"]),
                on="time",
                how="inner",
            ).sort("time")

        except ValidationError as e:
            raise ValueError(f"Invalid position input: {e!s}")

        # Calculate funding curve
        merged_df = merged_df.join(self._calculate_funding_curve(merged_df), on="time")

        # Prepare cache
        cache = self._prepare_cache(merged_df)

        # Calculate indicators
        results = self._calculate_indicators(cache)

        # Split results into two dictionaries
        dataframes = {k: v[k] for k, v in results["dataframes"].items()}
        indicators = {k: v["value"] for k, v in results["indicators"].items()}
        formatted_indicators = {k: v["formatted_value"] for k, v in results["indicators"].items()}

        # Create and return BacktestResult object
        return BacktestResult(
            _dataframes={"merged_df": merged_df, **dataframes},
            _indicator_values=indicators,
            _formatted_indicator_values=formatted_indicators,
        )

    def _prepare_cache(self, merged_df: pl.DataFrame) -> dict:
        """Prepare calculation cache

        Args:
            merged_df: DataFrame containing time, funding_curve and returns columns

        Returns:
            Dict containing cached data needed for calculations

        """
        times = merged_df.get_column("time")

        # Calculate total days
        total_days = (times[-1] - times[0]).total_seconds() / (24 * 3600)
        total_days = max(total_days, 1)  # Ensure at least 1 day

        # Calculate data frequency (in days)
        time_diffs = times.diff().drop_nulls()
        avg_interval = float(time_diffs.mean().total_seconds()) / (24 * 3600)

        # Calculate periods per day
        periods_per_day = 1 / avg_interval if avg_interval > 0 else 1

        return {
            "merged_df": merged_df,
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
                - commission: Commission rate
                - annual_trading_days: Number of trading days per year
                - indicators: List of indicators to calculate

        """
        return {
            "commission": self.commission,
            "annual_trading_days": self.annual_trading_days,
            "indicators": ("all" if self.indicators == "all" else list(self.sorted_indicators)),
        }
