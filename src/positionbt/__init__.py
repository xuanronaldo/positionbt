"""PositionBT is a simple, fast, and customizable backtesting framework
that directly evaluates trading strategies through position data (ranging from -1 to 1).
"""

from positionbt.core.backtester import PositionBacktester
from positionbt.indicators.base import BaseIndicator
from positionbt.indicators.registry import indicator_registry
from positionbt.models.models import BacktestResult
from positionbt.visualization.base import BaseFigure
from positionbt.visualization.registry import figure_registry
from positionbt.visualization.visualizer import BacktestVisualizer

__all__ = [
    "BacktestResult",
    "BacktestVisualizer",
    "BaseFigure",
    "BaseIndicator",
    "PositionBacktester",
    "figure_registry",
    "indicator_registry",
]
