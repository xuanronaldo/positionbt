import inspect

from positionbt.visualization import figures
from positionbt.visualization.base import BaseFigure

__all__ = ["figure_registry"]


class FigureRegistry:
    """Registry for managing visualization figures"""

    def __init__(self):
        self._registry: dict[str, type[BaseFigure]] = {}
        self._load_built_in_figures()

    def register(self, figure_cls: type[BaseFigure]) -> None:
        """Register a figure class

        Args:
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

        This method scans the figures module and registers all classes
        that inherit from BaseFigure (excluding BaseFigure itself)
        """
        # Get all members from figures module
        for name, obj in inspect.getmembers(figures):
            # Check if it's a class and inherits from BaseFigure
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseFigure)
                and obj != BaseFigure
                and hasattr(obj, "name")
            ):
                self.register(obj)


# Global figure registry instance
figure_registry = FigureRegistry()
