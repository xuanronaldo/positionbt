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

## Performance Test

To demonstrate the performance of PositionBT, we conducted tests using a real-world large-scale dataset:

### Test Dataset

* File: data/btc_ohlcv_1m.parquet
* Time Range: August 17, 2017 to December 22, 2024
* Data Type: 1-minute candlestick data
* Data Size: 2,682,218 records

### Test Environment

* Hardware: Mac Mini (M4 chip)
* Memory: 16 GB
* Storage: 256 GB

### Performance Results

* Backtest Duration: 0.14 seconds
* Data Processing Speed: 19,158,700 records/second

These test results demonstrate that PositionBT can efficiently process large-scale historical data, providing rapid feedback for strategy validation. Even when handling millions of data points, its performance remains stable.

> Note: Test results may vary depending on hardware configuration and data characteristics.

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
    commission_rate=0.001,  # 0.1% commission rate
    annual_trading_days=365,  # Use 365 trading days per year
    indicators="all",  # Calculate all available indicators
)

# Run backtest
backtest_result = backtester.run(position_df)

# Print backtest results in tabular format
backtest_result.print()

# Create visualizer and show results in browser
visualizer = BacktestVisualizer()
visualizer.show_in_browser(backtest_result, backtester.params)
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
    commission_rate=0.001,  # 0.1% commission rate
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

        @property
    def name(self) -> str:
        return "drawdown"

    @property
    def title(self) -> str:
        return "Strategy Drawdown"

    def create(self) -> go.Figure:
        """Create drawdown figure

        Returns:
            Plotly figure object containing drawdown visualization

        """
        # Calculate cumulative maximum of equity curve
        cummax = self.equity_curve.get_column("equity_curve").cum_max()

        # Calculate drawdown as percentage from peak
        drawdown = (self.equity_curve.get_column("equity_curve") - cummax) / cummax

        # Add drawdown trace to figure
        self._fig.add_trace(
            go.Scatter(
                x=self.equity_curve.get_column("time"),
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
    commission_rate=0.001,  # 0.1% commission rate
    annual_trading_days=365,  # Use 365 trading days per year
    indicators="all",  # Calculate all available indicators
)

# Run backtest
backtest_result = backtester.run(position_df)

# Create visualizer and show results in browser
visualizer = BacktestVisualizer(figures=["drawdown"])
visualizer.show_in_browser(backtest_result, backtester.params)
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
| Trading Performance | trading_performance | Shows overall trading performance | - Contains three subplots: equity curve, close price, and drawdown<br>- Marks trading points on equity curve and close price charts<br>- Uses different colors and shapes to mark long/short trades<br>- Supports interactive zoom viewing |
| Monthly Returns Distribution | monthly_returns | Shows monthly return distribution | - Uses bar chart for monthly returns<br>- Red/green colors distinguish profits/losses<br>- Supports precise return rate viewing<br>- Clearly displays monthly return distribution |
| Weekly Returns Heatmap | weekly_returns_heatmap | Shows weekly return distribution | - Similar to GitHub contribution graph heatmap display<br>- Uses red/blue color scheme to show profits/losses<br>- Supports precise weekly return viewing<br>- Intuitively displays return time distribution features |
| Position Analysis | position_analysis | Analyzes strategy position characteristics | - Contains three subplots: position frequency distribution, position return distribution, and position price return distribution<br>- Uses bar charts to show characteristics of different position ranges<br>- Supports interactive detailed data viewing<br>- Helps analyze the relationship between positions and returns |

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
    commission_rate: float = 0.0,
    annual_trading_days: int = 252,
    indicators: Union[str, list[str]] = "all"
)
```

**Parameters:**
- `close_df`: Polars DataFrame containing `time` and `close` columns
- `commission_rate`: Trading commission rate, default 0.0%
- `annual_trading_days`: Annual days, default 252
- `indicators`: Indicators to calculate, can be "all" or list of indicator names

**Main Methods:**
- `run(position_df: pl.DataFrame) -> BacktestResult`: Execute backtest and return results

#### BacktestResult
Data class containing all backtest results.

**Main Attributes:**
- `equity_curve`: NAV curve data
- `indicator_values`: Indicator calculation results
- `formatted_indicator_values`: Formatted indicator values

**Main Methods:**
- `print()`: Print backtest results in tabular format

#### BacktestVisualizer
Backtest result visualizer for generating interactive backtest reports.

```python
BacktestVisualizer(
    figures: Optional[list[str]] = None
)
```

**Parameters:**
- `figures`: List of figures to display (optional)

**Main Methods:**
- `show_in_browser(results: BacktestResult, params: dict, notes: Optional[str] = None)`: Show backtest report in browser
- `generate_html_report(results: BacktestResult, params: dict, output_path: str, notes: Optional[str] = None)`: Generate HTML format backtest report

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

**Required Methods:**
- `name(self) -> str`: Return chart unique identifier (requires @property decorator)
- `title(self) -> str`: Return chart display title (requires @property decorator)
- `create(self) -> go.Figure`: Create and return Plotly figure object

**Initialization Parameters:**
- `results`: BacktestResult object containing backtest data

**Available Attributes:**
- `results`: Backtest result object
- `equity_curve`: NAV curve data
- `_fig`: Base figure object (with default layout settings)

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