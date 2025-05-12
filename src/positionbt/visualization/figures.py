import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from positionbt.visualization.base import BaseFigure

# Define the order of figures
FIGURE_ORDER = [
    "TradingPerformanceFigure",
    "MonthlyReturnsFigure",
    "WeeklyReturnsHeatmap",
    "PositionAnalysisFigure",
]


class TradingPerformanceFigure(BaseFigure):
    """Trading performance visualization figure with equity curve, close price and drawdown subplots"""

    @property
    def name(self) -> str:
        return "trading_performance"

    @property
    def title(self) -> str:
        return "Trading Performance"

    def create(self) -> go.Figure:
        # Create subplots with shared X axis
        self._fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[1, 1, 0.5],
            subplot_titles=("Equity Curve", "Close Price", "Drawdown"),
        ).update_layout(height=1200)

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

        # Drawdown subplot (row 3)
        cummax = self.equity_curve.get_column("equity_curve").cum_max()
        drawdown = (self.equity_curve.get_column("equity_curve") - cummax) / cummax

        self._fig.add_trace(
            go.Scatter(
                x=self.equity_curve["time"],
                y=drawdown,
                fill="tozeroy",
                name="Drawdown",
                line=dict(color="red"),
            ),
            row=3,
            col=1,
        )

        # Get position data and find changes
        positions = merged_df.select(["time", "position"])
        position_changes = positions.with_columns(
            [
                pl.col("position").shift(1).fill_null(0).alias("prev_position"),
                pl.when(pl.col("position").diff().is_null()).then(pl.col("position")).otherwise(pl.col("position").diff()).alias("delta"),
                pl.col("position").sign().alias("current_dir"),
            ]
        ).filter(pl.col("delta").abs() > 1e-6)

        # Define visual encoding
        marker_config = {
            # Long positions
            (1, 1): dict(color="#4CAF50", symbol="triangle-up", label="Add Long"),
            (1, -1): dict(color="#4CAF50", symbol="triangle-down", label="Reduce Long"),
            # Short positions
            (-1, -1): dict(color="#F44336", symbol="triangle-down", label="Add Short"),
            (-1, 1): dict(color="#F44336", symbol="triangle-up", label="Reduce Short"),
            None: dict(color="#999", symbol="circle", label="Neutral Position"),
        }

        # Generate marker properties
        marker_data = []
        for time, pos, prev_position, delta, direction in position_changes.rows():
            action = 1 if delta > 0 else -1
            key = (direction, action)
            config = marker_config.get(key, marker_config[None])
            marker_data.append(
                {
                    "time": time,
                    "position": pos,
                    "prev_position": prev_position,
                    "color": config["color"],
                    "symbol": config["symbol"],
                    "label": config["label"],
                }
            )

        # Add markers to all subplots
        for row in [1, 2]:
            y_values = self.equity_curve["equity_curve"] if row == 1 else close_prices["close"] if row == 2 else drawdown
            marker_df = (
                pl.DataFrame(marker_data)
                .with_columns(pl.col("time").cast(merged_df.schema["time"]))
                .with_columns(pl.col("time").dt.replace_time_zone(merged_df.schema["time"].time_zone))
                .join(merged_df.select(["time", y_values.name]), on="time", how="inner")
            )

            hovertext = [
                f"""
                Time: {m['time']}<br>
                Prev Position: {m['prev_position']:.2f}<br>
                Position: {m['position']:.2f}<br>
                Status: {m['label']}
                """.replace("  ", "").replace("\n", "")
                for m in marker_data
            ]
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
                    hovertext=hovertext,
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

        # Update y-axis formatting for drawdown subplot
        self._fig.update_yaxes(
            tickformat=".1%",
            hoverformat=".2%",
            row=3,
            col=1,
        )

        # Update x-axis formatting for all subplots
        for row in [1, 2, 3]:
            self._fig.update_xaxes(
                tickformat="%Y-%m-%d",
                hoverformat="%Y-%m-%d %H:%M:%S",
                row=row,
                col=1,
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
            .agg((pl.col("returns") + 1).product().sub(1).alias("returns"))
            .sort("month")
        )

        # Add bar trace for monthly returns
        self._fig.add_trace(
            go.Bar(
                x=monthly_returns.get_column("month"),
                y=monthly_returns.get_column("returns"),
                name="Monthly Returns",
                marker_color=["red" if x < 0 else "green" for x in monthly_returns.get_column("returns")],
            )
        )

        # Update layout
        self._fig.update_layout(
            title=dict(
                text="Monthly Returns Distribution",
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
                font=dict(size=16),
            ),
            xaxis=dict(
                title="Month",
                gridcolor="#e9ecef",
                zerolinecolor="#e9ecef",
            ),
            yaxis=dict(
                title="Returns",
                tickformat=".1%",
                hoverformat=".2%",
                gridcolor="#e9ecef",
                zerolinecolor="#e9ecef",
            ),
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=50, l=50, r=50, b=50),
            showlegend=False,
        )

        return self._fig


class WeeklyReturnsHeatmap(BaseFigure):
    """Weekly returns heatmap figure similar to GitHub contribution graph"""

    @property
    def name(self) -> str:
        return "weekly_returns_heatmap"

    @property
    def title(self) -> str:
        return "Weekly Returns Heatmap"

    def create(self) -> go.Figure:
        """Create weekly returns heatmap figure

        Returns:
            Plotly figure object containing weekly returns heatmap visualization
        """
        # Calculate weekly returns
        weekly_returns = (
            self.equity_curve.with_columns(
                [
                    pl.col("equity_curve").pct_change().alias("returns"),
                    pl.col("time").dt.strftime("%Y-%m-%d").alias("date"),
                    pl.col("time").dt.week().alias("week"),
                    pl.col("time").dt.year().alias("year"),
                ]
            )
            .group_by(["year", "week"])
            .agg(
                [
                    (pl.col("returns") + 1).product().sub(1).alias("returns"),
                    pl.col("date").min().alias("start_date"),
                    pl.col("date").max().alias("end_date"),
                ]
            )
            .sort(["year", "week"])
        )

        # Create heatmap data
        years = weekly_returns.get_column("year").unique()
        weeks = list(range(1, 54))

        # Initialize heatmap data with zeros
        heatmap_data = np.zeros((len(years), 53))
        hover_text = np.empty((len(years), 53), dtype=object)

        # Fill heatmap data
        for year, week, returns, start_date, end_date in weekly_returns.rows():
            year_idx = list(years).index(year)
            week_idx = week - 1
            heatmap_data[year_idx, week_idx] = returns
            hover_text[year_idx, week_idx] = f"Year: {year}<br>Week: {week}<br>Period: {start_date} to {end_date}<br>Returns: {returns:.2%}"

        # Create heatmap
        self._fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data,
                x=weeks,
                y=years,
                text=hover_text,
                hoverinfo="text",
                colorscale="RdBu",
                reversescale=True,
                zmid=0,
                zmin=-weekly_returns.get_column("returns").abs().max(),
                zmax=weekly_returns.get_column("returns").abs().max(),
                colorbar=dict(
                    title="Returns",
                    tickformat=".1%",
                    thickness=10,
                    len=0.5,
                    y=0.5,
                ),
            )
        )

        # Update layout
        self._fig.update_layout(
            title=dict(
                text="Weekly Returns Heatmap",
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
                font=dict(size=16),
            ),
            xaxis=dict(
                title="Week of Year",
                tickmode="array",
                ticktext=[f"W{i}" for i in range(1, 54)],
                tickvals=weeks,
                gridcolor="#e9ecef",
                zerolinecolor="#e9ecef",
            ),
            yaxis=dict(
                title="Year",
                autorange="reversed",
                gridcolor="#e9ecef",
                zerolinecolor="#e9ecef",
            ),
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=50, l=50, r=50, b=50),
        )

        return self._fig


class PositionAnalysisFigure(BaseFigure):
    """Position analysis figure"""

    @property
    def name(self) -> str:
        return "position_analysis"

    @property
    def title(self) -> str:
        return "Position Analysis"

    def create(self) -> go.Figure:
        """Create position analysis figure with three histograms:
        1. Position frequency distribution
        2. Position net returns distribution
        3. Position price returns distribution

        Returns:
            Plotly figure object containing position analysis visualizations
        """
        # Get trade records
        trade_records = self.results.get_dataframe("trade_records")

        # Create subplots
        self._fig = make_subplots(
            rows=1,
            cols=3,
            horizontal_spacing=0.1,
            subplot_titles=(
                "Position Frequency Distribution",
                "Position Return Rate Distribution",
                "Position Price Return Distribution",
            ),
            column_widths=[0.33, 0.33, 0.34],
        )

        # Define bins for position analysis
        bins = np.arange(-1, 1.0, 0.25)
        # Create labels for each bin (including the last bin)
        bin_labels = [f"[{b:.2f}, {b + 0.25:.2f})" for b in bins]
        # Add one more label for the last bin
        bin_labels.append("[1.0, 1.0]")

        # Calculate position frequency
        position_counts = trade_records.group_by(pl.col("position").cut(bins, labels=bin_labels)).agg(pl.len().alias("count")).sort("position")

        # Add position frequency histogram
        self._fig.add_trace(
            go.Bar(
                x=position_counts.get_column("position"),
                y=position_counts.get_column("count"),
                name="Position Frequency",
                marker_color="#1f77b4",
            ),
            row=1,
            col=1,
        )

        # Calculate average net returns for each position bin
        position_returns = (
            trade_records.group_by(pl.col("position").cut(bins, labels=bin_labels))
            .agg(pl.col("return_rate").mean().alias("avg_returns"))
            .sort("position")
        )

        # Add position returns histogram
        self._fig.add_trace(
            go.Bar(
                x=position_returns.get_column("position"),
                y=position_returns.get_column("avg_returns"),
                name="Average Returns",
                marker_color="#2ca02c",
            ),
            row=1,
            col=2,
        )

        # Calculate average price returns for each position bin
        position_price_returns = (
            trade_records.group_by(pl.col("position").cut(bins, labels=bin_labels))
            .agg(pl.col("price_return").mean().alias("avg_price_returns"))
            .sort("position")
        )

        # Add position price returns histogram
        self._fig.add_trace(
            go.Bar(
                x=position_price_returns.get_column("position"),
                y=position_price_returns.get_column("avg_price_returns"),
                name="Average Price Returns",
                marker_color="#ff7f0e",
            ),
            row=1,
            col=3,
        )

        # Update layout
        self._fig.update_layout(
            height=500,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=50, l=50, r=50, b=50),
            showlegend=False,
        )

        # Update x-axis for all subplots
        for col in [1, 2, 3]:
            self._fig.update_xaxes(
                title="Position Range",
                gridcolor="#e9ecef",
                zerolinecolor="#e9ecef",
                row=1,
                col=col,
            )

        # Update y-axis for position frequency
        self._fig.update_yaxes(
            title="Frequency",
            gridcolor="#e9ecef",
            zerolinecolor="#e9ecef",
            row=1,
            col=1,
        )

        # Update y-axis for position returns
        self._fig.update_yaxes(
            title="Average Returns",
            tickformat=".1%",
            hoverformat=".2%",
            gridcolor="#e9ecef",
            zerolinecolor="#e9ecef",
            row=1,
            col=2,
        )

        # Update y-axis for position price returns
        self._fig.update_yaxes(
            title="Average Price Returns",
            tickformat=".1%",
            hoverformat=".2%",
            gridcolor="#e9ecef",
            zerolinecolor="#e9ecef",
            row=1,
            col=3,
        )

        return self._fig
