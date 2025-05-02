from datetime import datetime

import polars as pl
import pytest

from positionbt.core.backtester import PositionBacktester


@pytest.fixture
def sample_data():
    # Create sample data for testing
    dates = [datetime(2023, 1, i) for i in range(1, 6)]
    close_data = {"time": dates, "close": [100.0, 101.0, 99.0, 102.0, 103.0]}
    position_data = {"time": dates, "position": [0.0, 1.0, 1.0, -1.0, 0.0]}
    return {
        "close_df": pl.DataFrame(close_data),
        "position_df": pl.DataFrame(position_data),
    }


def test_backtester_initialization(sample_data):
    # Test basic initialization
    bt = PositionBacktester(sample_data["close_df"])
    assert bt.commission == 0.0
    assert bt.annual_trading_days == 252


def test_backtester_invalid_commission():
    # Test invalid commission rate
    with pytest.raises(ValueError):
        PositionBacktester(pl.DataFrame(), commission=-0.1)


def test_backtester_invalid_trading_days():
    # Test invalid annual trading days
    with pytest.raises(ValueError):
        PositionBacktester(pl.DataFrame(), annual_trading_days=0)


def test_calculate_equity_curve(sample_data):
    # Test equity curve calculation
    bt = PositionBacktester(sample_data["close_df"], commission=0.001)
    result = bt.run(sample_data["position_df"])

    # Verify basic properties of equity curve
    assert "equity_curve" in result.dataframes["merged_df"].columns
    assert "net_returns" in result.dataframes["merged_df"].columns


def test_pandas_input_compatibility(sample_data):
    # Test pandas DataFrame input compatibility
    pd_close_df = sample_data["close_df"].to_pandas()
    pd_position_df = sample_data["position_df"].to_pandas()

    bt = PositionBacktester(pd_close_df)
    result = bt.run(pd_position_df)
    assert result is not None


def test_params_property(sample_data):
    # Test parameters property
    bt = PositionBacktester(
        sample_data["close_df"],
        commission=0.002,
        annual_trading_days=250,
        indicators=["sharpe_ratio"],
    )
    params = bt.params
    assert params["commission"] == 0.002
    assert params["annual_trading_days"] == 250
    assert "sharpe_ratio" in params["indicators"]
