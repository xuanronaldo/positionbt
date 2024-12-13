from abc import ABC, abstractmethod

import plotly.graph_objects as go

from positionbt.models.models import BacktestResult


class BaseFigure(ABC):
    """Base class for visualization figures"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def title(self) -> str:
        pass

    def __init__(self, results: BacktestResult):
        """Initialize base figure

        Args:
            results: Backtest result object containing data for visualization

        """
        self.results = results
        self.funding_curve = results.funding_curve
        self._fig = self._create_base_figure()

    @abstractmethod
    def create(self) -> go.Figure:
        """Create the figure

        This method must be implemented by subclasses to create
        their specific visualization.

        Returns:
            Plotly figure object

        """
        pass

    def _create_base_figure(self) -> go.Figure:
        """Create base figure object with common settings

        Returns:
            Plotly figure object with default layout settings

        """
        fig = go.Figure()
        fig.update_layout(
            title=self.title,
            template="plotly_white",
            showlegend=True,
            hovermode="x unified",
            plot_bgcolor="white",
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Microsoft YaHei"),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128,128,128,0.2)",
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128,128,128,0.2)",
            ),
        )
        return fig
