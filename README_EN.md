# PositionBT

PositionBT is a Python library focused on position-based strategy backtesting. The design philosophy is "simplicity is beauty", aiming to provide a lightweight yet powerful backtesting framework.
> Note: The current version only supports single-instrument backtesting. Multi-instrument backtesting functionality is under development and will be available in future releases.

## Features

* Easy to Use
  - Unlike traditional backtesting frameworks that deal with complex orders, fees, and slippage models, PositionBT uses standardized position data for backtesting, greatly simplifying the workflow.
  - Positions use a standardized design with intuitive scales from -1 (full short) to 1 (full long), making strategy evaluation clearer and more straightforward.
  - Focus on validating core strategy logic, avoiding excessive time spent on trading execution details, improving strategy development efficiency.

* High Performance
  - Built on Polars, providing ultra-fast data processing capabilities.
  - Optimized calculation process, avoiding redundant computations, improving backtesting efficiency.

* Visualization
  - Integrated with `plotly` to create professional visualization modules, offering rich interactive analysis features.
  - Supports multiple viewing methods: preview backtest reports in real-time through browsers, or export as HTML files for sharing and archiving.

* Extensibility
  - Flexible indicator system: supports custom performance evaluation indicators.
  - Rich visualization options: supports custom chart types, enabling multi-dimensional analysis displays such as strategy performance curves, drawdown analysis, and position distribution.
  - Open interface design: facilitates user extension and integration of their own analysis tools to create personalized backtesting analysis workflows.

## Backtesting Speed

Taking the Bitcoin (BTC) closing price dataset in the `data` directory as an example. This dataset contains 15-minute level data from August 17, 2017, to December 11, 2024, with a total of 256,058 records. Running a simple "buy and hold" strategy on an M4 chip Mac Mini, the backtest calculation takes only 0.13 seconds, generating an interactive HTML report takes about 0.896 seconds, and the average backtesting time per 100,000 rows of data is just 0.051 seconds, demonstrating excellent performance.

## Installation

```bash
pip install positionbt
```

## Quick Start

### Buy and Hold Strategy Example

#### Code

Below is a simple strategy backtesting example: maintaining a full long position in BTC throughout the entire backtesting period. This example demonstrates the basic usage of PositionBT, including the complete process of data loading, backtest execution, and result visualization.

```python
import polars as pl

from examples.data_loader import load_close_data
from positionbt import BacktestVisualizer, PositionBacktester

# Load BTC close data
close_df = load_close_data()

# Generate position data
position_df = close_df.select(pl.col("time")).with_columns(pl.lit(1).alias("position"))

# Initialize backtester with parameters
backtester = PositionBacktester(
    close_df=close_df,
    commission=0.001,  # 0.1% commission rate
    annual_trading_days=365,  # Use 365 trading days per year
    indicators="all",  # Calculate all available indicators
)

# Run backtest
backtest_result = backtester.run(position_df)

# Print backtest results in tabular format
backtest_result.print()

# Create visualizer and show results in browser
visualizer = BacktestVisualizer(backtest_result, backtester.params)
visualizer.show_in_browser()
```

#### Results

Backtest Results:

![Backtest Results Preview](./docs/images/backtest_result.png)

Backtest Report:

![Backtest Report Preview](./docs/images/backtest_report.png)

### Custom Indicator Example

#### Code

PositionBT supports custom indicators, allowing users to add their own indicators for backtesting. Custom indicators will be directly displayed when printing backtest results or outputting backtest reports.

```python
import polars as pl

from examples.data_loader import load_close_data
from positionbt import BaseIndicator, PositionBacktester, indicator_registry


class MonthlyReturn(BaseIndicator):
    """Monthly return indicator"""

    @property
    def name(self) -> str:
        return "monthly_return"

    @property
    def requires(self) -> set[str]:
        # Depends on annual return
        return {"annual_return"}

    def calculate(self, cache: dict) -> float:
        """Calculate monthly return

        Calculation method:
        1. Convert from annual return
        2. Using formula: (1 + r_annual)^(1/12) - 1

        Args:
            cache: Dictionary containing calculation cache

        Returns:
            Monthly return value

        """
        if "monthly_return" not in cache:
            annual_return = cache["annual_return"]
            monthly_return = (1 + annual_return) ** (1 / 12) - 1
            cache["monthly_return"] = monthly_return

        return cache["monthly_return"]

    def format(self, value: float) -> str:
        """Format monthly return value as percentage

        Args:
            value: Monthly return value

        Returns:
            Formatted string with percentage

        """
        return f"{value:.2%}"


# Register custom indicator
indicator_registry.register(MonthlyReturn())

# Load close data
close_df = load_close_data()

# Generate position data
position_df = close_df.select(pl.col("time")).with_columns(pl.lit(1).alias("position"))

# Create backtester instance (using all indicators including the newly registered monthly return)
backtester = PositionBacktester(
    close_df=close_df,
    commission=0.001,  # 0.1% commission rate
    annual_trading_days=365,  # Use 365 trading days per year
    indicators=["monthly_return"],  # Use all registered indicators
)

# Run backtest
results = backtester.run(position_df)

# Print results
results.print()
```

#### Results

Backtest Results:

![Custom Indicator Results](./docs/images/custom_indicator_result.png)

### Custom Visualization Example

#### Code

PositionBT supports custom visualization, allowing users to add their own visualization modules as needed.

```python
import plotly.graph_objects as go
import polars as pl

from examples.data_loader import load_close_data
from positionbt import (
    BacktestVisualizer,
    BaseFigure,
    PositionBacktester,
    figure_registry,
)


class DrawdownFigure(BaseFigure):
    """Drawdown visualization figure"""

    name = "drawdown"  # Unique identifier for the figure
    title = "Strategy Drawdown"  # Display title for the figure

    def create(self) -> go.Figure:
        """Create drawdown figure

        Returns:
            Plotly figure object containing drawdown visualization

        """
        # Calculate cumulative maximum of funding curve
        cummax = self.funding_curve.get_column("funding_curve").cum_max()

        # Calculate drawdown as percentage from peak
        drawdown = (self.funding_curve.get_column("funding_curve") - cummax) / cummax

        # Add drawdown trace to figure
        self._fig.add_trace(
            go.Scatter(
                x=self.funding_curve.get_column("time"),
                y=drawdown,
                fill="tozeroy",  # Fill area from line to zero
                name="drawdown",
                line=dict(color="red"),
            )
        )

        # Update layout with percentage formatting
        self._fig.update_layout(
            yaxis=dict(
                tickformat=".1%",  # Format y-axis ticks as percentages
                hoverformat=".2%",  # Format hover text as percentages
            ),
        )
        return self._fig


# Register custom figure
figure_registry.register(DrawdownFigure)

# Load close data
close_df = load_close_data()

# Generate position data
position_df = close_df.select(pl.col("time")).with_columns(pl.lit(1).alias("position"))


# Initialize backtester
backtester = PositionBacktester(
    close_df=close_df,
    commission=0.001,  # 0.1% commission rate
    annual_trading_days=365,  # Use 365 trading days per year
    indicators="all",  # Calculate all available indicators
)

# Run backtest
backtest_result = backtester.run(position_df)

# Create visualizer and show results in browser
visualizer = BacktestVisualizer(backtest_result, backtester.params, figures=["drawdown"])
visualizer.show_in_browser()
```

#### Results

Backtest Report:

![Custom Figure Results](./docs/images/custom_figure_result.png)

## Supported Indicators

| Indicator Name | Description | Calculation Method | Display Format | Indicator ID |
|----------------|-------------|-------------------|----------------|--------------|
| Total Return | Overall strategy return performance | Final NAV/Initial NAV - 1 | Percentage (xx.xx%) | total_return |
| Annual Return | Annualized return performance | (1 + Total Return)^(365/Actual Days) - 1 | Percentage (xx.xx%) | annual_return |
| Volatility | Annualized standard deviation of returns | Return Std Dev * √(Annualization Period) | Percentage (xx.xx%) | volatility |
| Sharpe Ratio | Risk-adjusted return metric | Annual Return/Annual Volatility | Decimal (xx.xx) | sharpe_ratio |
| Max Drawdown | Maximum NAV drawdown magnitude | Max((Historical High - Current NAV)/Historical High) | Percentage (xx.xx%) | max_drawdown |
| Max Drawdown Duration | Duration of maximum drawdown | Number of days in max drawdown period | xx Days | max_drawdown_duration |
| Win Rate | Proportion of profitable trades | Number of Profitable Trades/Total Trades | Percentage (xx.xx%) | win_rate |
| Avg Drawdown | Average of drawdowns | Arithmetic mean of all non-zero drawdowns | Percentage (xx.xx%) | avg_drawdown |
| Profit Loss Ratio | Ratio of average profit to average loss | \|Average Profit\|/\|Average Loss\| | Decimal (xx.xx) or ∞ | profit_loss_ratio |

> Note: All percentage indicators retain two decimal places, ratio indicators retain two decimal places.

## Supported Visualizations

## Backtest Report Components

### Chart Components

| Chart Name | Component ID | Description | Key Features |
|------------|--------------|-------------|--------------|
| NAV Curve | funding_curve | Shows strategy NAV changes | - Displays complete NAV trend<br>- Marks maximum drawdown period<br>- Marks peak and trough points of max drawdown<br>- Supports interactive zoom viewing |
| Monthly Returns Distribution | monthly_returns | Shows monthly return distribution | - Uses bar chart for monthly returns<br>- Red/green colors distinguish profits/losses<br>- Supports precise return rate viewing |

### Information Panels

| Panel Type | Display Content | Description |
|------------|----------------|-------------|
| Backtest Parameters | - Commission rate<br>- Annual trading days<br>- Used indicators | Shows basic backtest setup parameters |
| Data Information | - Start date<br>- End date<br>- Total days<br>- Data points<br>- Data frequency | Shows basic backtest data information |
| Performance Metrics | - Total return<br>- Annual return<br>- Sharpe ratio<br>- Maximum drawdown<br>- Other core metrics | Displays key strategy performance metrics in card format |

> Note: All chart components support interactive operations, including zoom, pan, and image export. Reports are generated in HTML format and can be viewed directly through browsers or saved as HTML files.

## API Reference

### Core Classes

#### PositionBacktester
Main class for the backtesting engine, used to execute strategy backtests.

```python
PositionBacktester(
    close_df: pl.DataFrame,
    commission: float = 0.001,
    annual_trading_days: int = 252,
    indicators: Union[str, list[str]] = "all"
)
```

**Parameters:**
- `close_df`: Polars DataFrame containing `time` and `close` columns
- `commission`: Trading commission rate, default 0.1%
- `annual_trading_days`: Annual days, default 252
- `indicators`: Indicators to calculate, can be "all" or list of indicator names

**Main Methods:**
- `run(position_df: pl.DataFrame) -> BacktestResult`: Execute backtest and return results

#### BacktestResult
Data class containing all backtest results.

**Main Attributes:**
- `funding_curve`: NAV curve data
- `indicator_values`: Indicator calculation results
- `formatted_indicator_values`: Formatted indicator values

**Main Methods:**
- `print()`: Print backtest results in tabular format

#### BacktestVisualizer
Backtest result visualizer for generating interactive backtest reports.

```python
BacktestVisualizer(
    results: BacktestResult,
    params: dict,
    template_path: Optional[Path] = None,
    figures: Optional[list[str]] = None,
    notes: Optional[str] = None
)
```

**Parameters:**
- `results`: Backtest result object
- `params`: Backtest parameter dictionary
- `template_path`: Custom template path (optional)
- `figures`: List of figures to display (optional)
- `notes`: Report notes (optional)

**Main Methods:**
- `show_in_browser()`: Show backtest report in browser
- `generate_html_report(output_path: str)`: Generate HTML format backtest report

### Base Classes

#### BaseIndicator
Base class for indicator calculation, used for custom performance indicators.

**Required Methods:**
- `name(self) -> str`: Return indicator name
- `calculate(self, cache: dict) -> float`: Calculate indicator value
- `format(self, value: float) -> str`: Format indicator value

**Optional Methods:**
- `requires(self) -> set[str]`: Required indicators

#### BaseFigure
Base class for visualization charts, used for custom chart types.

**Required Attributes:**
- `name`: Unique chart identifier
- `title`: Chart display title

**Required Methods:**
- `create(self) -> go.Figure`: Create and return Plotly figure object

### Registries

#### indicator_registry
Indicator registry for managing and accessing available performance indicators.

**Main Methods:**
- `register(indicator_cls: Type[BaseIndicator])`: Register new indicator
- `get(name: str) -> Type[BaseIndicator]`: Get indicator class
- `available_indicators`: Get list of all available indicators

#### figure_registry
Chart registry for managing and accessing available visualization charts.

**Main Methods:**
- `register(figure_cls: Type[BaseFigure])`: Register new chart
- `get(name: str) -> Type[BaseFigure]`: Get chart class
- `available_figures`: Get list of all available charts 