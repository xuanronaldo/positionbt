from datetime import datetime, timedelta

import polars as pl
import pytest

from positionbt.core.backtester import PositionBacktester
from positionbt.indicators.base import BaseIndicator


class CustomIndicator(BaseIndicator):
    """Custom indicator for testing purposes"""

    def name(self):
        return "custom_indicator"

    def calculate(self, cache):
        net_returns = cache["merged_df"].get_column("net_returns")
        return float(net_returns.mean())

    def format(self, value):
        return f"{value:.4f}"


@pytest.fixture
def large_sample_data():
    # Create large sample dataset
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(252)]
    close_prices = [100 * (1 + i * 0.001) for i in range(252)]
    positions = [1.0 if i % 2 == 0 else -1.0 for i in range(252)]

    close_data = {"time": dates, "close": close_prices}
    position_data = {"time": dates, "position": positions}
    return {
        "close_df": pl.DataFrame(close_data),
        "position_df": pl.DataFrame(position_data),
    }


def test_full_backtest_workflow(large_sample_data):
    # Test complete backtesting workflow
    bt = PositionBacktester(
        large_sample_data["close_df"],
        commission=0.001,
        indicators=["sharpe_ratio", "max_drawdown"],
    )

    # Add custom indicator
    bt.add_indicator(CustomIndicator())

    # Run backtest
    result = bt.run(large_sample_data["position_df"])

    # Verify results
    assert result.dataframes is not None
    assert "sharpe_ratio" in result.indicator_values
    assert "max_drawdown" in result.indicator_values
    assert result.formatted_indicator_values is not None


def test_multiple_runs_consistency(large_sample_data):
    # Test consistency across multiple runs
    bt = PositionBacktester(large_sample_data["close_df"])

    result1 = bt.run(large_sample_data["position_df"])
    result2 = bt.run(large_sample_data["position_df"])

    # Verify results are consistent
    assert (
        result1.dataframes["merged_df"].get_column("equity_curve")
        == result2.dataframes["merged_df"].get_column("equity_curve")
    ).all()


def test_edge_cases(large_sample_data):
    bt = PositionBacktester(large_sample_data["close_df"])

    # Test empty positions
    zero_positions = pl.DataFrame(
        {
            "time": large_sample_data["position_df"]["time"],
            "position": [0.0] * len(large_sample_data["position_df"]),
        }
    )
    result = bt.run(zero_positions)

    # Verify equity curve stays at 1 for empty positions
    equity_curve = result.dataframes["merged_df"].get_column("equity_curve")
    assert (equity_curve == 1.0).all()


def test_high_frequency_data(large_sample_data):
    # Create high frequency data
    dates = [datetime(2023, 1, 1) + timedelta(minutes=i) for i in range(1000)]
    close_prices = [100 * (1 + i * 0.0001) for i in range(1000)]
    positions = [1.0 if i % 2 == 0 else -1.0 for i in range(1000)]

    hf_close_df = pl.DataFrame({"time": dates, "close": close_prices})
    hf_position_df = pl.DataFrame({"time": dates, "position": positions})

    bt = PositionBacktester(hf_close_df, commission=0.0001)
    result = bt.run(hf_position_df)

    # Verify high frequency data processing
    assert len(result.dataframes["merged_df"]) == 1000
