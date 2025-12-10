import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
from typing import Dict, Any, List, Optional, Tuple
import os
from app.core.config import settings
from app.utils.helpers import get_document_path


class VisualizationService:
    """Service for creating data visualizations from documents"""

    def __init__(self):
        self.supported_chart_types = {
            "bar": self._create_bar_chart,
            "line": self._create_line_chart,
            "pie": self._create_pie_chart,
            "scatter": self._create_scatter_chart,
            "histogram": self._create_histogram,
            "box": self._create_box_plot,
            "heatmap": self._create_heatmap,
        }

    def create_visualization(
        self,
        document_id: str,
        filename: str,
        chart_type: str = "bar",
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a visualization from a document"""
        try:
            # Get document path
            file_path = get_document_path(document_id, filename)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Document not found: {file_path}")

            # Load data based on file type
            df = self._load_data_for_visualization(file_path)

            if df is None or df.empty:
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in [".txt", ".md", ".pdf", ".docx"]:
                    raise ValueError(
                        "Visualization is not available for text documents. Text documents can only be analyzed for insights and statistics."
                    )
                else:
                    raise ValueError("No data available for visualization")

            # Apply filters if provided
            if filters:
                df = self._apply_filters(df, filters)

            # Select columns if specified
            if columns:
                available_columns = [col for col in columns if col in df.columns]
                if available_columns:
                    df = df[available_columns]

            # Create the chart
            if chart_type not in self.supported_chart_types:
                chart_type = "bar"  # Default fallback

            chart_func = self.supported_chart_types[chart_type]
            fig, insights = chart_func(df, columns)

            # Convert to JSON-serializable format
            chart_data = json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))

            return {
                "chart_data": chart_data,
                "chart_type": chart_type,
                "insights": insights,
                "data_shape": df.shape,
                "columns": list(df.columns),
            }

        except Exception as e:
            raise Exception(f"Error creating visualization: {str(e)}")

    def _load_data_for_visualization(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from file for visualization"""
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext == ".csv":
                return pd.read_csv(file_path)
            elif file_ext in [".xlsx", ".xls"]:
                return pd.read_excel(file_path)
            elif file_ext == ".json":
                with open(file_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    return pd.DataFrame([data])
                else:
                    return None
            else:
                # Try to read as CSV as fallback
                try:
                    return pd.read_csv(file_path)
                except:
                    return None
        except Exception as e:
            print(f"Error loading data for visualization: {e}")
            return None

    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to dataframe"""
        filtered_df = df.copy()

        for column, filter_value in filters.items():
            if column in filtered_df.columns:
                if isinstance(filter_value, dict):
                    # Range filter
                    if "min" in filter_value:
                        filtered_df = filtered_df[
                            filtered_df[column] >= filter_value["min"]
                        ]
                    if "max" in filter_value:
                        filtered_df = filtered_df[
                            filtered_df[column] <= filter_value["max"]
                        ]
                elif isinstance(filter_value, list):
                    # List filter
                    filtered_df = filtered_df[filtered_df[column].isin(filter_value)]
                else:
                    # Exact match
                    filtered_df = filtered_df[filtered_df[column] == filter_value]

        return filtered_df

    def _create_bar_chart(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Tuple[go.Figure, List[str]]:
        """Create a bar chart"""
        insights = []

        if len(df.columns) < 2:
            # Single column - count occurrences
            if (
                df.columns[0]
                in df.select_dtypes(include=["object", "category"]).columns
            ):
                value_counts = df[df.columns[0]].value_counts().head(20)
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Distribution of {df.columns[0]}",
                    labels={"x": df.columns[0], "y": "Count"},
                )
                insights.append(f"Showing top 20 values for {df.columns[0]}")
            else:
                fig = px.bar(
                    x=df.index,
                    y=df[df.columns[0]],
                    title=f"Values of {df.columns[0]}",
                    labels={"x": "Index", "y": df.columns[0]},
                )
        else:
            # Multiple columns - use first two
            x_col = df.columns[0]
            y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

            if df[y_col].dtype in ["int64", "float64"]:
                fig = px.bar(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} by {x_col}",
                    labels={"x": x_col, "y": y_col},
                )
                insights.append(f"Average {y_col}: {df[y_col].mean():.2f}")
            else:
                # Categorical y-axis
                fig = px.bar(
                    df,
                    x=x_col,
                    color=y_col,
                    title=f"Distribution by {x_col} and {y_col}",
                    labels={"x": x_col},
                )

        return fig, insights

    def _create_line_chart(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Tuple[go.Figure, List[str]]:
        """Create a line chart"""
        insights = []

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        if len(numeric_cols) >= 2:
            x_col = df.columns[0]
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]

            fig = px.line(
                df, x=x_col, y=y_col, title=f"{y_col} over {x_col}", markers=True
            )
            insights.append(
                f"Trend shows {y_col} ranging from {df[y_col].min():.2f} to {df[y_col].max():.2f}"
            )
        elif len(numeric_cols) == 1:
            fig = px.line(
                df,
                x=df.index,
                y=numeric_cols[0],
                title=f"{numeric_cols[0]} Trend",
                markers=True,
            )
            insights.append(f"Single numeric column trend visualization")
        else:
            # Fallback to bar chart
            fig, insights = self._create_bar_chart(df, columns)

        return fig, insights

    def _create_pie_chart(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Tuple[go.Figure, List[str]]:
        """Create a pie chart"""
        insights = []

        # Find categorical column
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        if len(cat_cols) > 0:
            col = cat_cols[0]
            value_counts = df[col].value_counts().head(10)  # Limit to top 10

            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {col}",
            )
            insights.append(f"Showing top 10 categories for {col}")
            insights.append(f"Total categories: {len(df[col].unique())}")
        else:
            # Use first numeric column
            num_cols = df.select_dtypes(include=["int64", "float64"]).columns
            if len(num_cols) > 0:
                col = num_cols[0]
                # Create bins for pie chart
                df["bins"] = pd.cut(df[col], bins=5)
                value_counts = df["bins"].value_counts()

                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution of {col} (binned)",
                )
                insights.append(f"Numeric data binned into 5 categories")
            else:
                # Fallback
                fig = go.Figure()
                fig.add_annotation(
                    text="No suitable data for pie chart", showarrow=False
                )
                insights.append("No categorical or numeric data found")

        return fig, insights

    def _create_scatter_chart(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Tuple[go.Figure, List[str]]:
        """Create a scatter plot"""
        insights = []

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]

            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}",
                trendline="ols" if len(df) > 10 else None,
            )

            # Calculate correlation
            corr = df[x_col].corr(df[y_col])
            insights.append(f"Correlation coefficient: {corr:.3f}")

            if abs(corr) > 0.7:
                insights.append("Strong correlation detected")
            elif abs(corr) > 0.3:
                insights.append("Moderate correlation detected")
            else:
                insights.append("Weak correlation detected")

        else:
            fig = go.Figure()
            fig.add_annotation(
                text="Need at least 2 numeric columns for scatter plot", showarrow=False
            )
            insights.append("Insufficient numeric columns for scatter plot")

        return fig, insights

    def _create_histogram(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Tuple[go.Figure, List[str]]:
        """Create a histogram"""
        insights = []

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        if len(numeric_cols) > 0:
            col = numeric_cols[0]

            fig = px.histogram(df, x=col, title=f"Distribution of {col}", nbins=30)

            insights.append(f"Mean: {df[col].mean():.2f}")
            insights.append(f"Median: {df[col].median():.2f}")
            insights.append(f"Std Dev: {df[col].std():.2f}")
        else:
            fig = go.Figure()
            fig.add_annotation(text="No numeric columns for histogram", showarrow=False)
            insights.append("No numeric data available")

        return fig, insights

    def _create_box_plot(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Tuple[go.Figure, List[str]]:
        """Create a box plot"""
        insights = []

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        if len(numeric_cols) > 0:
            col = numeric_cols[0]

            fig = px.box(df, y=col, title=f"Box Plot of {col}")

            insights.append(f"Median: {df[col].median():.2f}")
            insights.append(
                f"IQR: {df[col].quantile(0.75) - df[col].quantile(0.25):.2f}"
            )
            insights.append(
                f"Outliers: {((df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))) | (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))).sum()}"
            )
        else:
            fig = go.Figure()
            fig.add_annotation(text="No numeric columns for box plot", showarrow=False)
            insights.append("No numeric data available")

        return fig, insights

    def _create_heatmap(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Tuple[go.Figure, List[str]]:
        """Create a correlation heatmap"""
        insights = []

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()

            fig = px.imshow(
                corr_matrix, title="Correlation Heatmap", text_auto=True, aspect="auto"
            )

            insights.append(
                "Correlation heatmap shows relationships between numeric variables"
            )
            insights.append(
                f"Strongest correlation: {corr_matrix.abs().max().max():.3f}"
            )
        else:
            fig = go.Figure()
            fig.add_annotation(
                text="Need at least 2 numeric columns for heatmap", showarrow=False
            )
            insights.append("Insufficient numeric columns for correlation heatmap")

        return fig, insights
