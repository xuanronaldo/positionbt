import os
import tempfile
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

from positionbt.models.models import BacktestResult
from positionbt.visualization.base import BaseFigure
from positionbt.visualization.registry import figure_registry


class BacktestVisualizer:
    """Backtest result visualizer"""

    def __init__(
        self,
        results: BacktestResult,
        params: dict,
        template_path: Optional[Path] = None,
        figures: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ):
        """Initialize visualizer

        Args:
            results: BacktestResult object
            params: Dictionary of backtest parameters
            template_path: Path to HTML template file
            figures: List of figure names to display. If empty list, no figures will be shown.
                    If None, all registered figures will be shown.
            notes: Optional notes to display in the report

        """
        self.results = results
        self.params = params
        self.notes = notes
        self._template_path = template_path or (
            Path(__file__).parent.parent / "visualization" / "templates" / "report_template.html"
        )

        # Initialize figure dictionary
        self.figures: dict[str, BaseFigure] = {}

        # Add specified figures
        if figures is not None:  # figures parameter explicitly specified
            if figures:  # if figures is not an empty list
                for name in figures:
                    figure_cls = figure_registry.get(name)
                    self.figures[name] = figure_cls(results)
        else:  # figures is None, show all registered figures
            for name in figure_registry.available_figures:
                figure_cls = figure_registry.get(name)
                self.figures[name] = figure_cls(results)

    def _generate_all_figures(self) -> dict[str, str]:
        """Generate HTML code for all figures

        Returns:
            Dictionary mapping figure names to their HTML representations

        """
        figures_html = {}

        # Generate figures
        for name, figure in self.figures.items():
            figure_html = f"""
            <div class="chart">
                <h3>{figure.title}</h3>
                {figure.create().to_html(full_html=False, include_plotlyjs=True)}
            </div>
            """
            figures_html[name] = figure_html

        return figures_html

    def _generate_backtest_params_html(self) -> str:
        """Generate HTML for backtest parameters

        Returns:
            HTML string containing formatted backtest parameters

        """
        params = {
            "Commission Rate": f"{self.params['commission']:.3%}",
            "Annual Trading Days": f"{self.params['annual_trading_days']} days",
            "Indicators": (
                "All indicators"
                if self.params["indicators"] == "all"
                else ", ".join(self.params["indicators"])
            ),
        }

        return "\n".join(
            f'<div class="info-item"><span style="color: #666;">{k}:</span> {v}</div>'
            for k, v in params.items()
        )

    def _generate_data_info_html(self) -> str:
        """Generate HTML for data information

        Returns:
            HTML string containing formatted data information

        """
        df = self.results.funding_curve
        start_date = df.select(pl.col("time").min()).item().strftime("%Y-%m-%d")
        end_date = df.select(pl.col("time").max()).item().strftime("%Y-%m-%d")
        total_days = (
            df.select(pl.col("time").max()).item() - df.select(pl.col("time").min()).item()
        ).days

        # Calculate data frequency
        time_diff = df.select(pl.col("time").diff().median()).item()
        minutes = time_diff.total_seconds() / 60

        if minutes < 60:  # Less than 1 hour
            frequency = f"{minutes:.0f}min"
        elif minutes < 1440:  # Less than 1 day
            frequency = f"{minutes/60:.1f}h"
        elif minutes < 10080:  # Less than 1 week
            frequency = f"{minutes/1440:.1f}d"
        elif minutes < 43200:  # Less than 1 month
            frequency = f"{minutes/10080:.1f}w"
        else:  # Greater than or equal to 1 month
            frequency = f"{minutes/43200:.1f}m"

        info = {
            "Start Date": start_date,
            "End Date": end_date,
            "Total Days": f"{total_days} days",
            "Data Points": f"{len(df):,}",
            "Data Frequency": frequency,
        }

        return "\n".join(
            f'<div class="info-item"><span style="color: #666;">{k}:</span> {v}</div>'
            for k, v in info.items()
        )

    def _generate_metrics_html(self) -> str:
        """Generate HTML for metrics

        Returns:
            HTML string containing formatted metrics

        """
        metrics_html = ""

        # Get formatted metric values from BacktestResult object
        for key, value in self.results.formatted_indicator_values.items():
            # Convert underscore-separated keys to title case display names
            display_name = " ".join(word.capitalize() for word in key.split("_"))

            metrics_html += f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-name">{display_name}</div>
            </div>
            """

        return metrics_html

    def _generate_notes_html(self) -> str:
        """Generate HTML for notes section

        Returns:
            HTML string containing formatted notes, or empty string if no notes

        """
        if not self.notes:
            return ""

        return f"""
        <div class="info-section">
            <h2>Notes</h2>
            <div class="info-content">
                <div class="info-item">{self.notes}</div>
            </div>
        </div>
        """

    def generate_html_report(self, output_path: str) -> None:
        """Generate HTML backtest report

        Args:
            output_path: Path where the HTML report will be saved

        """
        # Read HTML template
        with open(self._template_path, encoding="utf-8") as f:
            html_template = f.read()

        # Generate all figure HTML code
        figures_html = self._generate_all_figures()

        # Prepare template variables
        template_vars = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "backtest_params_html": self._generate_backtest_params_html(),
            "data_info_html": self._generate_data_info_html(),
            "metrics_html": self._generate_metrics_html(),
            "notes_html": self._generate_notes_html(),
            "figures": figures_html,
        }

        # Replace template variables
        html_content = html_template
        for key, value in template_vars.items():
            if key != "figures":
                placeholder = f"${key}"
                html_content = html_content.replace(placeholder, str(value))

        # Find and replace figures placeholder
        figures_placeholder = "$figures"
        if figures_placeholder in html_content:
            figures_section = "\n".join(html for html in figures_html.values())
            html_content = html_content.replace(figures_placeholder, figures_section)

        # Save HTML file
        with open(output_path, "w", encoding="utf-8", errors="xmlcharrefreplace") as f:
            f.write(html_content)

    def show_in_browser(self, delay: float = 0.5) -> None:
        """Display backtest results in browser

        Args:
            delay: Delay in seconds before cleaning up temporary file

        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", encoding="utf-8", delete=False
        ) as tmp_file:
            # Generate report and write to temporary file
            self.generate_html_report(tmp_file.name)
            # Get temporary file path
            tmp_path = Path(tmp_file.name)
            # Open file in browser
            webbrowser.open(f"file://{tmp_path.absolute()}")

            time.sleep(delay)
            try:
                os.unlink(tmp_path)
            except Exception as e:
                print(f"Warning: Failed to clean up temporary file: {e}")
