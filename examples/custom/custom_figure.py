import plotly.graph_objects as go
import polars as pl

from examples.data_loader import load_close_data
from positionbt import (
    BacktestVisualizer,
    BaseFigure,
    PositionBacktester,
    figure_registry,
)


class DrawdownFigure(BaseFigure):
    """Drawdown visualization figure"""

    @property
    def name(self) -> str:
        return "drawdown"

    @property
    def title(self) -> str:
        return "Strategy Drawdown"

    def create(self) -> go.Figure:
        """Create drawdown figure

        Returns:
            Plotly figure object containing drawdown visualization

        """
        # Calculate cumulative maximum of equity curve
        cummax = self.equity_curve.get_column("equity_curve").cum_max()

        # Calculate drawdown as percentage from peak
        drawdown = (self.equity_curve.get_column("equity_curve") - cummax) / cummax

        # Add drawdown trace to figure
        self._fig.add_trace(
            go.Scatter(
                x=self.equity_curve.get_column("time"),
                y=drawdown,
                fill="tozeroy",  # Fill area from line to zero
                name="drawdown",
                line=dict(color="red"),
            )
        )

        # Update layout with percentage formatting
        self._fig.update_layout(
            yaxis=dict(
                tickformat=".1%",  # Format y-axis ticks as percentages
                hoverformat=".2%",  # Format hover text as percentages
            ),
        )
        return self._fig


# Register custom figure
figure_registry.register(DrawdownFigure)

# Load close data
close_df = load_close_data()

# Generate position data
position_df = close_df.select(pl.col("time")).with_columns(pl.lit(1).alias("position"))


# Initialize backtester
backtester = PositionBacktester(
    close_df=close_df,
    commission_rate=0.001,  # 0.1% commission rate
    annual_trading_days=365,  # Use 365 trading days per year
    indicators="all",  # Calculate all available indicators
)

# Run backtest
backtest_result = backtester.run(position_df)

# Print results
backtest_result.print()

# Create visualizer and show results in browser
visualizer = BacktestVisualizer()
visualizer.show_in_browser(backtest_result, backtester.params)
