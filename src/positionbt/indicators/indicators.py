import polars as pl

from positionbt.indicators.base import BaseIndicator


class TotalReturn(BaseIndicator):
    """Total return indicator"""

    @property
    def name(self) -> str:
        return "total_return"

    def calculate(self, cache: dict) -> float:
        """Calculate total return

        Args:
            cache: Dictionary containing calculation cache

        Returns:
            Total return as a float

        """
        if "total_return" not in cache:
            curve = cache["merged_df"].get_column("equity_curve")
            cache["total_return"] = round(float(curve.tail(1)[0] / curve[0] - 1), 3)
        return cache["total_return"]

    def format(self, value: float) -> str:
        """Format total return value as percentage

        Args:
            value: Total return value

        Returns:
            Formatted string with percentage

        """
        return f"{value:.2%}"


class AnnualReturn(BaseIndicator):
    """Annualized return indicator"""

    @property
    def name(self) -> str:
        return "annual_return"

    @property
    def requires(self) -> set[str]:
        """Dependencies on other indicators

        Returns:
            Set containing "total_return"

        """
        return {"total_return"}

    def calculate(self, cache: dict) -> float:
        """Calculate annualized return

        Args:
            cache: Dictionary containing calculation cache

        Returns:
            Annualized return as a float

        """
        if "annual_return" not in cache:
            total_return = cache["total_return"]
            total_days = cache["total_days"]

            cache["annual_return"] = round(float(((1 + total_return) ** (365 / total_days)) - 1), 3)
        return cache["annual_return"]

    def format(self, value: float) -> str:
        """Format annualized return value as percentage

        Args:
            value: Annualized return value

        Returns:
            Formatted string with percentage

        """
        return f"{value:.2%}"


class Volatility(BaseIndicator):
    """Volatility indicator"""

    @property
    def name(self) -> str:
        return "volatility"

    def calculate(self, cache: dict) -> float:
        """Calculate annualized volatility

        Args:
            cache: Dictionary containing calculation cache

        Returns:
            Annualized volatility as a float

        """
        net_returns = cache["merged_df"].get_column("net_returns")
        periods_per_day = cache["periods_per_day"]
        annual_periods = periods_per_day * cache["annual_trading_days"]

        cache["volatility"] = round(net_returns.std() * (annual_periods**0.5), 3)
        return cache["volatility"]

    def format(self, value: float) -> str:
        """Format volatility value as percentage

        Args:
            value: Volatility value

        Returns:
            Formatted string with percentage

        """
        return f"{value:.2%}"


class SharpeRatio(BaseIndicator):
    """Sharpe ratio indicator"""

    @property
    def name(self) -> str:
        return "sharpe_ratio"

    @property
    def requires(self) -> set[str]:
        """Dependencies on other indicators

        Returns:
            Set containing "annual_return" and "volatility"

        """
        return {"annual_return", "volatility"}

    def calculate(self, cache: dict) -> float:
        """Calculate Sharpe ratio

        Args:
            cache: Dictionary containing calculation cache

        Returns:
            Sharpe ratio as a float

        """
        if "sharpe_ratio" not in cache:
            if "volatility" not in cache:
                volatility_indicator = Volatility()
                cache["volatility"] = volatility_indicator.calculate(cache)

            annual_vol = cache["volatility"]
            cache["sharpe_ratio"] = round(float(cache["annual_return"] / annual_vol if annual_vol != 0 else 0), 3)

        return cache["sharpe_ratio"]

    def format(self, value: float) -> str:
        """Format Sharpe ratio value

        Args:
            value: Sharpe ratio value

        Returns:
            Formatted string with two decimal places

        """
        return f"{value:.2f}"


class MaxDrawdown(BaseIndicator):
    """Maximum drawdown indicator"""

    @property
    def name(self) -> str:
        return "max_drawdown"

    def calculate(self, cache: dict) -> float:
        """Calculate maximum drawdown

        Calculation method:
        1. Calculate historical peak at each point
        2. Calculate drawdown from peak at each point
        3. Find the maximum drawdown value

        Args:
            cache: Dictionary containing calculation cache

        Returns:
            Maximum drawdown as a float

        """
        if "max_drawdown" not in cache:
            df = cache["merged_df"]

            # Calculate historical peak
            df = df.with_columns(pl.col("equity_curve").cum_max().alias("peak"))

            # Calculate drawdown
            df = df.with_columns(((pl.col("peak") - pl.col("equity_curve")) / pl.col("peak")).alias("drawdown"))

            # Get maximum drawdown
            cache["max_drawdown"] = round(float(df.get_column("drawdown").max()), 3)

        return cache["max_drawdown"]

    def format(self, value: float) -> str:
        """Format maximum drawdown value as percentage

        Args:
            value: Maximum drawdown value

        Returns:
            Formatted string with percentage

        """
        return f"{value:.2%}"


class MaxDrawdownDuration(BaseIndicator):
    """Maximum drawdown duration indicator"""

    @property
    def name(self) -> str:
        return "max_drawdown_duration"

    @property
    def requires(self) -> set[str]:
        """Dependencies on other indicators

        Returns:
            Set containing "max_drawdown"

        """
        return {"max_drawdown"}

    def calculate(self, cache: dict) -> float:
        """Calculate maximum drawdown duration in days

        Args:
            cache: Dictionary containing calculation cache

        Returns:
            Maximum drawdown duration in days as a float

        """
        df = cache["merged_df"]

        # Calculate historical peak
        df = df.with_columns(pl.col("equity_curve").cum_max().alias("peak"))

        # Calculate drawdown
        df = df.with_columns(((pl.col("peak") - pl.col("equity_curve")) / pl.col("peak")).alias("drawdown"))

        # Find the end time of the maximum drawdown
        max_drawdown_idx = df.get_column("drawdown").arg_max()
        max_drawdown_end = df.get_column("time")[max_drawdown_idx]

        # Find the start time of the drawdown (most recent peak)
        peak_before_max_dd = df.filter(pl.col("time") <= max_drawdown_end).filter(pl.col("equity_curve") == pl.col("peak")).get_column("time")[-1]

        # Calculate duration in days
        duration_seconds = (max_drawdown_end - peak_before_max_dd).total_seconds()
        duration_days = round(duration_seconds / (24 * 3600), 0)

        cache["max_drawdown_duration"] = duration_days
        return duration_days

    def format(self, value: float) -> str:
        """Format maximum drawdown duration as days

        Args:
            value: Maximum drawdown duration in days

        Returns:
            Formatted string with number of days

        """
        return f"{value:.0f} days"


class WinRate(BaseIndicator):
    """Win rate indicator"""

    @property
    def name(self) -> str:
        return "win_rate"

    def calculate(self, cache: dict) -> float:
        """Calculate win rate

        Args:
            cache: Dictionary containing calculation cache

        Returns:
            Win rate as a float

        """
        return_rate = cache["trade_records"].get_column("return_rate")
        total_trades = len(return_rate)
        if total_trades == 0:
            return 0.0
        winning_trades = (return_rate > 0).sum()
        cache["win_rate"] = round(winning_trades / total_trades, 3)
        return cache["win_rate"]

    def format(self, value: float) -> str:
        """Format win rate value as percentage

        Args:
            value: Win rate value

        Returns:
            Formatted string with percentage

        """
        return f"{value:.2%}"


class AvgDrawdown(BaseIndicator):
    """Average drawdown indicator"""

    @property
    def name(self) -> str:
        return "avg_drawdown"

    def calculate(self, cache: dict) -> float:
        """Calculate average drawdown

        Calculation method:
        1. Calculate drawdown at each point
        2. Consider only non-zero drawdowns
        3. Calculate the average of these drawdowns

        Args:
            cache: Dictionary containing calculation cache

        Returns:
            Average drawdown as a float

        """
        if "avg_drawdown" not in cache:
            curve = cache["merged_df"].get_column("equity_curve")

            # Calculate historical peak
            peak = curve.cum_max()

            # Calculate drawdown
            drawdown = (peak - curve) / peak

            # Consider only non-zero drawdowns
            non_zero_drawdown = drawdown.filter(drawdown > 0)

            # Calculate average drawdown
            cache["avg_drawdown"] = round(float(non_zero_drawdown.mean() if len(non_zero_drawdown) > 0 else 0), 3)

        return cache["avg_drawdown"]

    def format(self, value: float) -> str:
        """Format average drawdown value as percentage

        Args:
            value: Average drawdown value

        Returns:
            Formatted string with percentage

        """
        return f"{value:.2%}"


class ProfitLossRatio(BaseIndicator):
    """Profit-loss ratio indicator

    Calculation method: average profit / absolute value of average loss
    Profit-loss ratio = |average return of winning trades| / |average return of losing trades|
    """

    @property
    def name(self) -> str:
        return "profit_loss_ratio"

    def calculate(self, cache: dict) -> float:
        """Calculate profit-loss ratio

        Args:
            cache: Dictionary containing calculation cache

        Returns:
            Profit-loss ratio as a float

        """
        if "profit_loss_ratio" not in cache:
            return_rate = cache["trade_records"].get_column("return_rate")

            # Separate winning and losing trades
            profit_trades = return_rate.filter(return_rate > 0)
            loss_trades = return_rate.filter(return_rate < 0)

            # Calculate average profit and average loss
            avg_profit = profit_trades.mean() if len(profit_trades) > 0 else 0
            avg_loss = abs(loss_trades.mean()) if len(loss_trades) > 0 else 0

            # Calculate profit-loss ratio
            if avg_loss == 0:  # Avoid division by zero
                ratio = float("inf") if avg_profit > 0 else 0
            else:
                ratio = round(avg_profit / avg_loss, 3)

            cache["profit_loss_ratio"] = ratio

        return cache["profit_loss_ratio"]

    def format(self, value: float) -> str:
        """Format profit-loss ratio value

        Args:
            value: Profit-loss ratio value

        Returns:
            Formatted string with two decimal places or infinity symbol

        """
        if value == float("inf"):
            return "∞"
        return f"{value:.2f}"
