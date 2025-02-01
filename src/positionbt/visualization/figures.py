import plotly.graph_objects as go
import polars as pl

from positionbt.visualization.base import BaseFigure


class FundingCurveFigure(BaseFigure):
    """Funding curve visualization figure"""

    @property
    def name(self) -> str:
        return "funding_curve"

    @property
    def title(self) -> str:
        return "Funding Curve"

    def create(self) -> go.Figure:
        """Create funding curve figure

        Returns:
            Plotly figure object containing funding curve visualization

        """
        # Add funding curve trace
        self._fig.add_trace(
            go.Scatter(
                x=self.funding_curve.get_column("time"),
                y=self.funding_curve.get_column("funding_curve"),
                name="NAV",
                line=dict(color="#1f77b4"),
            )
        )

        # Check if max drawdown is available and add visualization
        if "max_drawdown" in self.results.indicator_values:
            cummax = self.funding_curve.get_column("funding_curve").cum_max()
            drawdown = (self.funding_curve.get_column("funding_curve") - cummax) / cummax
            self._add_max_drawdown_visualization(self._fig, cummax, drawdown)
        else:
            print("max_drawdown not found in indicators")  # Debug information

        # Update layout with specific y-axis formatting
        self._fig.update_layout(yaxis=dict(tickformat=".3f"))
        return self._fig

    def _add_max_drawdown_visualization(
        self, fig: go.Figure, cummax: pl.Series, drawdown: pl.Series
    ) -> None:
        """Add maximum drawdown visualization

        Args:
            fig: Plotly figure object
            cummax: Series of historical maximum values of the funding curve
            drawdown: Series of drawdown values

        """
        # Find start and end points of maximum drawdown
        max_dd_end_idx = drawdown.arg_min()
        max_dd_start_idx = (
            self.funding_curve.slice(0, max_dd_end_idx + 1).get_column("funding_curve").arg_max()
        )

        # Get data for drawdown region
        dd_region = self.funding_curve.slice(
            max_dd_start_idx, max_dd_end_idx - max_dd_start_idx + 1
        )

        # Add drawdown region trace
        fig.add_trace(
            go.Scatter(
                x=dd_region.get_column("time"),
                y=dd_region.get_column("funding_curve"),
                name=f"Max Drawdown Period ({self.results.indicator_values['max_drawdown']:.1%})",
                line=dict(color="rgba(255,0,0,0.5)"),
                showlegend=True,
            )
        )

        # Add fill area for drawdown
        fig.add_trace(
            go.Scatter(
                x=dd_region.get_column("time"),
                y=cummax.slice(max_dd_start_idx, max_dd_end_idx - max_dd_start_idx + 1),
                fill="tonexty",
                mode="none",
                fillcolor="rgba(255,0,0,0.2)",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Add markers for start and end of maximum drawdown
        for idx, name, color, symbol, position in [
            (max_dd_start_idx, "Peak", "green", "triangle-up", "top"),
            (max_dd_end_idx, "Trough", "red", "triangle-down", "bottom"),
        ]:
            point_value = self.funding_curve.row(idx)[1]  # funding_curve value
            fig.add_trace(
                go.Scatter(
                    x=[self.funding_curve.row(idx)[0]],  # time value
                    y=[point_value],
                    mode="markers+text",
                    name=name,
                    marker=dict(color=color, size=10, symbol=symbol),
                    text=[f"{name}: {point_value:.2f}"],
                    textposition=f"{position} center",
                )
            )


class MonthlyReturnsFigure(BaseFigure):
    """Monthly returns distribution figure"""

    @property
    def name(self) -> str:
        return "monthly_returns"

    @property
    def title(self) -> str:
        return "Monthly Returns Distribution"

    def create(self) -> go.Figure:
        """Create monthly returns distribution figure

        Returns:
            Plotly figure object containing monthly returns distribution visualization

        """
        # Calculate monthly returns
        monthly_returns = (
            self.funding_curve.with_columns(
                [
                    pl.col("funding_curve").pct_change().alias("returns"),
                    pl.col("time").dt.strftime("%Y-%m").alias("month"),
                ]
            )
            .group_by("month")
            .agg(pl.col("returns").sum())
            .sort("month")
        )

        # Add bar trace for monthly returns
        self._fig.add_trace(
            go.Bar(
                x=monthly_returns.get_column("month"),
                y=monthly_returns.get_column("returns"),
                name="Monthly Returns",
                marker_color=[
                    "red" if x < 0 else "green" for x in monthly_returns.get_column("returns")
                ],
            )
        )

        # Update layout with specific y-axis formatting
        self._fig.update_layout(
            yaxis=dict(
                tickformat=".1%",
                hoverformat=".2%",
            ),
        )
        return self._fig
