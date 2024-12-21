"""Data validation utilities"""

from typing import Literal, Union

import pandas as pd
import polars as pl

from positionbt.core.constants import REQUIRED_COLUMNS
from positionbt.indicators.registry import indicator_registry


class ValidationError(Exception):
    """Exception raised for errors in the data validation process."""

    pass


def validate_commission(commission: float) -> None:
    """Validate commission rate

    Args:
        commission: Commission rate as a float

    Raises:
        ValidationError: If commission is not a number or not between 0 and 1

    """
    if not isinstance(commission, (int, float)):
        raise ValidationError("Commission must be a number")
    if commission < 0 or commission > 1:
        raise ValidationError("Commission must be between 0 and 1")


def validate_annual_trading_days(days: int) -> None:
    """Validate annual trading days

    Args:
        days: Number of trading days in a year

    Raises:
        ValidationError: If days is not a positive integer or exceeds 365

    """
    if not isinstance(days, int):
        raise ValidationError("Annual trading days must be an integer")
    if days <= 0:
        raise ValidationError("Annual trading days must be positive")
    if days > 365:
        raise ValidationError("Annual trading days cannot exceed 365")


def validate_indicators(indicators: Union[str, list[str]]) -> list[str]:
    """Validate indicator parameters and return sorted list of indicators including dependencies

    Args:
        indicators: Indicator name or list of names

    Returns:
        List[str]: Sorted list of indicators including all dependencies

    Raises:
        ValidationError: If indicators parameter is invalid

    """
    if indicators != "all" and not isinstance(indicators, (list, tuple)):
        raise ValidationError("Indicators must be 'all' or a list of indicator names")

    sorted_indicators = indicator_registry.sorted_indicators

    if indicators == "all":
        return sorted_indicators

    # Validate all indicators are available
    invalid_indicators = set(indicators) - set(indicator_registry.available_indicators)
    if invalid_indicators:
        raise ValidationError(
            f"Invalid indicator names: {list(invalid_indicators)}. "
            f"Available indicators: {indicator_registry.available_indicators}"
        )

    # Collect all required indicators (including dependencies)
    required_indicators = set(indicators)
    pending_indicators = list(indicators)

    while pending_indicators:
        indicator = pending_indicators.pop()
        indicator_cls = indicator_registry.get_indicator(indicator)
        if len(indicator_cls.requires) > 0:
            for required in indicator_cls.requires:
                if required not in required_indicators:
                    required_indicators.add(required)
                    pending_indicators.append(required)

    # Return specified indicators (including dependencies) in registry order
    return [name for name in sorted_indicators if name in required_indicators]


def validate_and_convert_input(
    df: Union[pl.DataFrame, pd.DataFrame],
    data_type: Literal["close", "position"],
) -> pl.DataFrame:
    """Validate and convert input data

    Args:
        df: Input DataFrame
        data_type: Data type, either 'close' or 'position'

    Returns:
        pl.DataFrame: Validated and converted Polars DataFrame

    Raises:
        ValidationError: If data does not meet requirements

    """
    try:
        # Convert to Polars DataFrame
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)

        # Validate required columns
        required_cols = REQUIRED_COLUMNS[data_type]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")

        # Validate time column type
        if not pl.Series(df["time"]).dtype.is_temporal():
            raise ValidationError("Time column must be datetime type")

        # Validate time sorting
        if not pl.Series(df["time"]).is_sorted():
            df = df.sort("time")

        # Validate position value range (if position data)
        if data_type == "position":
            position_values = df["position"]
            if (position_values < -1).any() or (position_values > 1).any():
                raise ValidationError("Position values must be between -1 and 1")

        return df

    except Exception as e:
        if not isinstance(e, ValidationError):
            raise ValidationError(f"Data validation failed: {e!s}")
        raise


def validate_time_alignment(close_df: pl.DataFrame, position_df: pl.DataFrame) -> None:
    """Validate time alignment

    Args:
        close_df: Price data DataFrame
        position_df: Position data DataFrame

    Raises:
        ValidationError: If timestamps are not aligned

    """
    if not close_df["time"].equals(position_df["time"]):
        raise ValidationError("Close and position data must have identical timestamps")


def validate_data_type(data) -> Union[float, pl.DataFrame]:
    """Validate data type is either float or polars.DataFrame

    Args:
        data: Data to validate

    Returns:
        Union[float, pl.DataFrame]: Validated data

    Raises:
        ValidationError: If data type is invalid

    """
    if isinstance(data, (float, pl.DataFrame)):
        return data
    raise ValidationError("Data must be of type float or polars.DataFrame")
