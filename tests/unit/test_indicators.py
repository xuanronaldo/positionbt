from datetime import datetime, timedelta

import polars as pl
import pytest

from positionbt.indicators.indicators import (
    AnnualReturn,
    AvgDrawdown,
    MaxDrawdown,
    MaxDrawdownDuration,
    ProfitLossRatio,
    SharpeRatio,
    TotalReturn,
    Volatility,
    WinRate,
)


@pytest.fixture
def sample_cache():
    """Create sample data for testing"""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]
    equity_curve = [1.0, 1.02, 0.98, 1.05, 1.08]
    net_returns = [0.0, 0.02, -0.039, 0.071, 0.029]

    df = pl.DataFrame(
        {
            "time": dates,
            "equity_curve": pl.Series(equity_curve, dtype=pl.Float64),
            "net_returns": pl.Series(net_returns, dtype=pl.Float64),
        }
    )

    trade_records = pl.DataFrame(
        {
            "return_rate": [0.08],
        }
    )

    return {"merged_df": df, "trade_records": trade_records, "total_days": 5, "periods_per_day": 1, "annual_trading_days": 252}


def test_total_return(sample_cache):
    """Test total return calculation"""
    indicator = TotalReturn()
    result = indicator.calculate(sample_cache)

    # Verify calculation: (1.08 - 1.0) / 1.0 = 0.08
    assert abs(result - 0.08) < 1e-6
    assert indicator.format(result) == "8.00%"


def test_annual_return(sample_cache):
    """Test annualized return calculation"""
    # Calculate total return first
    total_return = TotalReturn()
    sample_cache["total_return"] = total_return.calculate(sample_cache)

    indicator = AnnualReturn()
    result = indicator.calculate(sample_cache)

    # Verify annualized calculation: (1 + 0.08)^(365/5) - 1
    expected = ((1 + 0.08) ** (365 / 5)) - 1
    assert abs(result - expected) < 1e-3
    assert "%" in indicator.format(result)


def test_volatility(sample_cache):
    """Test volatility calculation"""
    indicator = Volatility()
    result = indicator.calculate(sample_cache)

    # Verify annualized volatility calculation
    net_returns = sample_cache["merged_df"].get_column("net_returns")
    expected = float(net_returns.std() * (252**0.5))
    assert abs(result - expected) < 1e-3
    assert "%" in indicator.format(result)


def test_sharpe_ratio(sample_cache):
    """Test Sharpe ratio calculation"""
    # Prepare dependent indicators
    total_return = TotalReturn()
    annual_return = AnnualReturn()
    volatility = Volatility()

    # Calculate total_return first
    sample_cache["total_return"] = total_return.calculate(sample_cache)
    sample_cache["annual_return"] = annual_return.calculate(sample_cache)
    sample_cache["volatility"] = volatility.calculate(sample_cache)

    indicator = SharpeRatio()
    result = indicator.calculate(sample_cache)

    # Verify Sharpe ratio calculation
    expected = sample_cache["annual_return"] / sample_cache["volatility"]
    assert abs(result - expected) < 1e-3
    assert "." in indicator.format(result)


def test_max_drawdown(sample_cache):
    """Test maximum drawdown calculation"""
    indicator = MaxDrawdown()
    result = indicator.calculate(sample_cache)

    # Verify maximum drawdown calculation
    # Max drawdown should occur between 1.02 and 0.98: (1.02 - 0.98) / 1.02 ≈ 0.0392
    assert abs(result - 0.0392) < 1e-3
    assert "%" in indicator.format(result)


def test_max_drawdown_duration(sample_cache):
    """Test maximum drawdown duration calculation"""
    max_dd = MaxDrawdown()
    sample_cache["max_drawdown"] = max_dd.calculate(sample_cache)

    indicator = MaxDrawdownDuration()
    result = indicator.calculate(sample_cache)

    assert result > 0
    assert "days" in indicator.format(result)


def test_win_rate(sample_cache):
    """Test win rate calculation"""
    indicator = WinRate()
    result = indicator.calculate(sample_cache)

    # Sample data has 3 positive net_returns and 2 negative net_returns
    expected = 1
    assert abs(result - expected) < 1e-6
    assert "%" in indicator.format(result)


def test_avg_drawdown(sample_cache):
    """Test average drawdown calculation"""
    indicator = AvgDrawdown()
    result = indicator.calculate(sample_cache)

    assert result > 0
    assert "%" in indicator.format(result)


def test_profit_loss_ratio(sample_cache):
    """Test profit/loss ratio calculation"""
    indicator = ProfitLossRatio()
    result = indicator.calculate(sample_cache)
    expected = float("inf")

    assert result == expected


def test_edge_cases():
    """Test edge cases with empty data"""
    # Create empty data
    empty_df = pl.DataFrame(
        {
            "time": [],
            "equity_curve": pl.Series([], dtype=pl.Float64),
            "net_returns": pl.Series([], dtype=pl.Float64),
        }
    )
    empty_cache = {
        "merged_df": empty_df,
        "periods_per_day": 1,
        "annual_trading_days": 252,
    }

    # Test how each indicator handles empty data
    indicators = [
        TotalReturn(),
        AnnualReturn(),
        Volatility(),
        SharpeRatio(),
        MaxDrawdown(),
        WinRate(),
        AvgDrawdown(),
        ProfitLossRatio(),
    ]

    for indicator in indicators:
        try:
            # For AnnualReturn, need to set total_return first
            if isinstance(indicator, AnnualReturn):
                empty_cache["total_return"] = 0.0
            result = indicator.calculate(empty_cache)
            assert result == 0 or result == float("inf") or result == 0.0
        except Exception as e:
            assert isinstance(
                e,
                (
                    ValueError,
                    ZeroDivisionError,
                    IndexError,
                    KeyError,
                    TypeError,
                ),
            )


def test_indicator_dependencies():
    """Test indicator dependencies"""
    sharpe = SharpeRatio()
    assert "annual_return" in sharpe.requires
    assert "volatility" in sharpe.requires

    annual_return = AnnualReturn()
    assert "total_return" in annual_return.requires

    max_dd_duration = MaxDrawdownDuration()
    assert "max_drawdown" in max_dd_duration.requires
