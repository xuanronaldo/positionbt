from abc import ABC, abstractmethod

import polars as pl


class BaseIndicator(ABC):
    """Base class for indicators"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Indicator name

        Returns:
            Unique identifier string for the indicator

        """
        pass

    @property
    def requires(self) -> set[str]:
        """Dependencies on other indicators

        Returns:
            Set of indicator names that this indicator depends on

        """
        return set()

    @abstractmethod
    def calculate(self, cache: dict) -> float | pl.DataFrame:
        """Calculate indicator value

        Args:
            cache: Dictionary containing calculation cache and intermediate results

        Returns:
            Either a float value or a DataFrame containing the calculation results

        """
        pass

    @abstractmethod
    def format(self, value: float) -> str:
        """Format indicator value

        Args:
            value: Raw indicator value to be formatted

        Returns:
            Formatted string representation of the value

        """
        pass
