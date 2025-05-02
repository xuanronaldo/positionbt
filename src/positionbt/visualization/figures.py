import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from positionbt.visualization.base import BaseFigure


class EquityCurveFigure(BaseFigure):
    """equity curve visualization figure with close price subplot"""

    @property
    def name(self) -> str:
        return "equity_curve"

    @property
    def title(self) -> str:
        return "Equity Curve"

    def create(self) -> go.Figure:
        # Create subplots with shared X axis
        self._fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[1, 1]
        ).update_layout(height=1000)

        # Get close price data
        merged_df = self.results.get_dataframe("merged_df")
        close_prices = merged_df.select(["time", "close"])

        # Main equity curve (row 1)
        self._fig.add_trace(
            go.Scatter(
                x=self.equity_curve["time"],
                y=self.equity_curve["equity_curve"],
                name="Equity Curve",
                line=dict(color="#1f77b4"),
            ),
            row=1,
            col=1,
        )

        # Close price subplot (row 2)
        self._fig.add_trace(
            go.Scatter(
                x=close_prices["time"],
                y=close_prices["close"],
                name="Close Price",
                line=dict(color="#2ca02c"),
            ),
            row=2,
            col=1,
        )

        # Get position data and find changes
        positions = merged_df.select(["time", "position"])
        position_changes = positions.with_columns(
            [
                pl.col("position").diff().alias("delta"),
                pl.col("position").sign().alias("current_dir"),
            ]
        ).filter(pl.col("delta").abs() > 1e-6)

        # Define visual encoding
        marker_config = {
            # Long positions
            (1, 1): dict(color="#4CAF50", symbol="triangle-up", label="Add Long"),  # Green ▲
            (1, -1): dict(color="#4CAF50", symbol="triangle-down", label="Reduce Long"),  # Green ▼
            # Short positions
            (-1, -1): dict(color="#F44336", symbol="triangle-down", label="Add Short"),  # Red ▼
            (-1, 1): dict(color="#F44336", symbol="triangle-up", label="Reduce Short"),  # Red ▲
            None: dict(color="#999", symbol="circle", label="Neutral Position"),
        }

        # Generate marker properties
        marker_data = []
        for time, pos, delta, direction in position_changes.rows():
            action = 1 if delta > 0 else -1
            key = (direction, action)
            config = marker_config.get(key, marker_config[None])
            marker_data.append(
                {
                    "time": time,
                    "position": pos,
                    "color": config["color"],
                    "symbol": config["symbol"],
                    "label": config["label"],
                }
            )

        # Add markers to both subplots
        for row in [1, 2]:
            y_values = self.equity_curve["equity_curve"] if row == 1 else close_prices["close"]
            marker_df = (
                pl.DataFrame(marker_data)
                .with_columns(pl.col("time").cast(merged_df.schema["time"]))
                .with_columns(
                    pl.col("time").dt.replace_time_zone(merged_df.schema["time"].time_zone)
                )
                .join(merged_df.select(["time", y_values.name]), on="time", how="inner")
            )

            self._fig.add_trace(
                go.Scatter(
                    x=marker_df["time"],
                    y=marker_df[y_values.name],
                    mode="markers",
                    marker=dict(
                        symbol=marker_df["symbol"],
                        size=10,
                        color=marker_df["color"],
                        line=dict(width=1, color="white"),
                    ),
                    hoverinfo="text",
                    hovertext=[
                        f"Time: {m['time']}<br>Position: {m['position']:.2f}<br>Status: {m['label']}"
                        for m in marker_data
                    ],
                    showlegend=False,
                ),
                row=row,
                col=1,
            )

        # Add legend annotation
        self._fig.add_annotation(
            x=0.98,
            y=1.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            text=(
                "<span style='color:#4CAF50'>▲</span> Add Long | "
                "<span style='color:#4CAF50'>▼</span> Reduce Long<br>"
                "<span style='color:#F44336'>▼</span> Add Short | "
                "<span style='color:#F44336'>▲</span> Reduce Short<br>"
                "<span style='color:#999'>●</span> Neutral Position"
            ),
            bgcolor="rgba(255,255,255,0.9)",
        )

        return self._fig


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
            self.equity_curve.with_columns(
                [
                    pl.col("equity_curve").pct_change().alias("returns"),
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
