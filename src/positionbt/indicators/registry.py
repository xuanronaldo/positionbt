import inspect

from positionbt.indicators import indicators
from positionbt.indicators.base import BaseIndicator

__all__ = ["indicator_registry"]


class IndicatorRegistry:
    """Registry for managing indicators"""

    def __init__(self):
        self._indicators: dict[str, type[BaseIndicator]] = {}
        self._register_builtin_indicators()

    def _register_builtin_indicators(self) -> None:
        """Automatically register all built-in indicators

        This method scans the indicators module and registers all classes
        that inherit from BaseIndicator
        """
        # Get all members from indicators module
        for name, obj in inspect.getmembers(indicators):
            # Check if it's a class and inherits from BaseIndicator
            if inspect.isclass(obj) and issubclass(obj, BaseIndicator) and obj != BaseIndicator:
                self.register(obj(), update_dependency=False)
        self._sort_indicators_by_dependency()

    def _sort_indicators_by_dependency(self) -> None:
        """Sort indicators based on their dependencies

        This method performs a topological sort to ensure indicators
        are calculated in the correct order based on their dependencies
        """
        sorted_indicators = []
        visited = set()

        def visit(indicator_name):
            if indicator_name in visited:
                return
            visited.add(indicator_name)
            indicator_class = self._indicators[indicator_name]
            # Get dependencies
            dependencies = indicator_class().requires
            for dep in dependencies:
                if dep in self._indicators:
                    visit(dep)
            sorted_indicators.append(indicator_name)

        for name in self._indicators:
            visit(name)

        # Update indicators order
        self._sorted_indicators = sorted_indicators

    def register(self, indicator: BaseIndicator, update_dependency: bool = True) -> None:
        """Register a new indicator

        Args:
            indicator: Instance of BaseIndicator to register
            update_dependency: Whether to update dependency sorting after registration

        """
        self._indicators[indicator.name] = indicator.__class__
        if update_dependency:
            self._sort_indicators_by_dependency()

    def get_indicator(self, name: str) -> BaseIndicator:
        """Get indicator instance by name

        Args:
            name: Name of the indicator to retrieve

        Returns:
            New instance of the requested indicator

        Raises:
            ValueError: If indicator name is not found in registry

        """
        if name not in self._indicators:
            raise ValueError(f"Unknown indicator: {name}")
        return self._indicators[name]()

    @property
    def available_indicators(self) -> list[str]:
        """Get list of all available indicator names

        Returns:
            Sorted list of registered indicator names

        """
        return sorted(str(key) for key in self._indicators.keys())

    @property
    def sorted_indicators(self) -> dict[str, type[BaseIndicator]]:
        """Get all indicators sorted by dependency

        Returns:
            List of indicator names in dependency order

        """
        return self._sorted_indicators


# Global indicator registry instance
indicator_registry = IndicatorRegistry()
