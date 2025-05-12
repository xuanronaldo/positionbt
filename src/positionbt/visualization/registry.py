import inspect
from collections import OrderedDict

from positionbt.visualization import figures
from positionbt.visualization.base import BaseFigure

__all__ = ["figure_registry"]


class FigureRegistry:
    """Registry for managing visualization figures"""

    def __init__(self):
        self._registry: OrderedDict[str, type[BaseFigure]] = OrderedDict()
        self._load_built_in_figures()

    def register(self, figure_cls: type[BaseFigure]) -> None:
        """Register a figure class

        Args:
            name: Name of the figure to register
            figure_cls: Figure class to register (must inherit from BaseFigure)

        Raises:
            ValueError: If figure class is not a subclass of BaseFigure

        """
        if not issubclass(figure_cls, BaseFigure):
            raise ValueError("Figure class must be a subclass of BaseFigure")
        self._registry[figure_cls.name] = figure_cls

    def get(self, name: str) -> type[BaseFigure]:
        """Get a figure class by name

        Args:
            name: Name of the figure to retrieve

        Returns:
            Figure class

        Raises:
            ValueError: If figure name is not found in registry

        """
        if name not in self._registry:
            raise ValueError(f"Figure '{name}' is not registered")
        return self._registry[name]

    @property
    def available_figures(self) -> list[str]:
        """Get list of all available figure names

        Returns:
            List of registered figure names

        """
        return list(self._registry.keys())

    def _load_built_in_figures(self) -> None:
        """Automatically load built-in figures

        This method loads figure classes in the order specified by FIGURE_ORDER
        """
        # Get all members from figures module
        members = dict(inspect.getmembers(figures))

        # Register classes in the order specified by FIGURE_ORDER
        for class_name in figures.FIGURE_ORDER:
            if class_name in members:
                obj = members[class_name]
                if inspect.isclass(obj) and issubclass(obj, BaseFigure) and obj != BaseFigure and hasattr(obj, "name"):
                    self.register(obj)


# Global figure registry instance
figure_registry = FigureRegistry()
