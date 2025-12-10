import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import os
import re
import json
from collections import Counter
from app.core.config import settings
from app.utils.helpers import get_document_path
from app.services.llm_service import LLMService


class AnalysisService:
    """Service for analyzing documents and providing insights"""

    def __init__(self):
        self.llm_service = None

    def _get_llm_service(self):
        if self.llm_service is None:
            self.llm_service = LLMService()
        return self.llm_service

    def analyze_document(
        self, document_id: str, filename: str, analysis_type: str = "summary"
    ) -> Dict[str, Any]:
        """Analyze a document and provide insights"""
        try:
            file_path = get_document_path(document_id, filename)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Document not found: {file_path}")

            file_ext = os.path.splitext(filename)[1].lower()

            if analysis_type == "summary":
                return self._generate_summary_analysis(file_path, file_ext, document_id)
            elif analysis_type == "statistics":
                return self._generate_statistics_analysis(
                    file_path, file_ext, document_id
                )
            elif analysis_type == "insights":
                return self._generate_insights_analysis(
                    file_path, file_ext, document_id
                )
            else:
                # Default to comprehensive analysis
                return self._generate_comprehensive_analysis(
                    file_path, file_ext, document_id
                )

        except Exception as e:
            raise Exception(f"Error analyzing document: {str(e)}")

    def _generate_summary_analysis(
        self, file_path: str, file_ext: str, document_id: str
    ) -> Dict[str, Any]:
        """Generate a summary analysis of the document"""
        try:
            if file_ext in [".csv", ".xlsx", ".xls"]:
                df = self._load_dataframe(file_path, file_ext)
                return self._analyze_dataframe_summary(df, document_id)
            else:
                # Text-based document
                text_content = self._extract_text_content(file_path, file_ext)
                return self._analyze_text_summary(text_content, document_id)

        except Exception as e:
            return {
                "document_id": document_id,
                "analysis_type": "summary",
                "summary": "Unable to generate summary due to processing error",
                "statistics": {},
                "insights": [f"Error: {str(e)}"],
                "recommendations": ["Check document format and try again"],
            }

    def _generate_statistics_analysis(
        self, file_path: str, file_ext: str, document_id: str
    ) -> Dict[str, Any]:
        """Generate statistical analysis of the document"""
        try:
            if file_ext in [".csv", ".xlsx", ".xls"]:
                df = self._load_dataframe(file_path, file_ext)
                return self._analyze_dataframe_statistics(df, document_id)
            else:
                text_content = self._extract_text_content(file_path, file_ext)
                return self._analyze_text_statistics(text_content, document_id)

        except Exception as e:
            return {
                "document_id": document_id,
                "analysis_type": "statistics",
                "summary": "Unable to generate statistics due to processing error",
                "statistics": {},
                "insights": [f"Error: {str(e)}"],
                "recommendations": ["Check document format and try again"],
            }

    def _generate_insights_analysis(
        self, file_path: str, file_ext: str, document_id: str
    ) -> Dict[str, Any]:
        """Generate insights analysis using LLM"""
        try:
            llm_svc = self._get_llm_service()

            if file_ext in [".csv", ".xlsx", ".xls"]:
                df = self._load_dataframe(file_path, file_ext)
                sample_data = df.head(10).to_string()
                prompt = f"""
                Analyze this dataset sample and provide key insights:

                Dataset Sample:
                {sample_data}

                Columns: {list(df.columns)}
                Shape: {df.shape}

                Please provide:
                1. Key patterns or trends you observe
                2. Potential data quality issues
                3. Recommendations for further analysis
                4. Business insights if applicable

                Be concise but insightful.
                """
            else:
                text_content = self._extract_text_content(file_path, file_ext)
                text_sample = text_content[:2000]  # Limit for LLM
                prompt = f"""
                Analyze this document content and provide key insights:

                Document Sample:
                {text_sample}

                Please provide:
                1. Main topics and themes
                2. Key information or findings
                3. Document structure and organization
                4. Potential areas for further exploration

                Be concise but insightful.
                """

            insights_response = llm_svc.generate_response(prompt)

            return {
                "document_id": document_id,
                "analysis_type": "insights",
                "summary": "AI-generated insights about the document",
                "statistics": {},
                "insights": [insights_response],
                "recommendations": [
                    "Consider deeper analysis of key findings",
                    "Validate insights with domain expertise",
                    "Explore related documents for context",
                ],
            }

        except Exception as e:
            return {
                "document_id": document_id,
                "analysis_type": "insights",
                "summary": "Unable to generate AI insights",
                "statistics": {},
                "insights": [f"Error generating insights: {str(e)}"],
                "recommendations": ["Try basic summary analysis instead"],
            }

    def _generate_comprehensive_analysis(
        self, file_path: str, file_ext: str, document_id: str
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis combining all types"""
        summary_analysis = self._generate_summary_analysis(
            file_path, file_ext, document_id
        )
        stats_analysis = self._generate_statistics_analysis(
            file_path, file_ext, document_id
        )

        # Combine insights
        all_insights = summary_analysis.get("insights", []) + stats_analysis.get(
            "insights", []
        )

        # Combine recommendations
        all_recommendations = list(
            set(
                summary_analysis.get("recommendations", [])
                + stats_analysis.get("recommendations", [])
            )
        )

        return {
            "document_id": document_id,
            "analysis_type": "comprehensive",
            "summary": f"{summary_analysis.get('summary', '')} {stats_analysis.get('summary', '')}",
            "statistics": {
                **summary_analysis.get("statistics", {}),
                **stats_analysis.get("statistics", {}),
            },
            "insights": all_insights,
            "recommendations": all_recommendations,
        }

    def _load_dataframe(self, file_path: str, file_ext: str) -> pd.DataFrame:
        """Load dataframe from file"""
        if file_ext == ".csv":
            return pd.read_csv(file_path)
        elif file_ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type for dataframe: {file_ext}")

    def _extract_text_content(self, file_path: str, file_ext: str) -> str:
        """Extract text content from file"""
        try:
            if file_ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            elif file_ext == ".md":
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            elif file_ext == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return json.dumps(data, indent=2)
            else:
                # For other formats, try to read as text
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
        except UnicodeDecodeError:
            # Fallback for binary files
            return f"Binary file ({file_ext}) - content analysis not available"

    def _analyze_dataframe_summary(
        self, df: pd.DataFrame, document_id: str
    ) -> Dict[str, Any]:
        """Analyze dataframe for summary"""
        insights = []
        recommendations = []

        # Basic info
        shape_info = f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns"

        # Data types
        dtype_counts = df.dtypes.value_counts()
        type_info = ", ".join(
            [f"{count} {dtype}" for dtype, count in dtype_counts.items()]
        )

        # Missing data
        missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_percent > 0:
            insights.append(f"Data completeness: {100 - missing_percent:.1f}%")
            if missing_percent > 10:
                recommendations.append("Consider handling missing data")

        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append(f"Found {len(numeric_cols)} numeric columns for analysis")
            recommendations.append("Consider visualization for numeric trends")

        # Categorical columns analysis
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            insights.append(f"Found {len(cat_cols)} categorical columns")
            unique_vals = sum(df[col].nunique() for col in cat_cols)
            insights.append(f"Total unique categories: {unique_vals}")

        return {
            "document_id": document_id,
            "analysis_type": "summary",
            "summary": f"{shape_info}. Data types: {type_info}.",
            "statistics": {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "data_completeness": f"{100 - missing_percent:.1f}%",
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(cat_cols),
            },
            "insights": insights,
            "recommendations": recommendations,
        }

    def _analyze_dataframe_statistics(
        self, df: pd.DataFrame, document_id: str
    ) -> Dict[str, Any]:
        """Generate statistical analysis for dataframe"""
        stats = {}
        insights = []
        recommendations = []

        # Numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe()

            for col in numeric_cols:
                col_stats = {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "missing": int(df[col].isnull().sum()),
                }
                stats[col] = col_stats

                # Generate insights
                if df[col].std() / df[col].mean() > 0.5:  # High variance
                    insights.append(f"High variance in {col} suggests diverse data")
                if df[col].isnull().sum() > 0:
                    insights.append(
                        f"{col} has {df[col].isnull().sum()} missing values"
                    )

        # Categorical statistics
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            value_counts = df[col].value_counts()
            top_values = value_counts.head(5).to_dict()

            stats[f"{col}_categories"] = {
                "unique_values": int(df[col].nunique()),
                "top_values": top_values,
                "missing": int(df[col].isnull().sum()),
            }

            # Insights for categorical
            if df[col].nunique() == 1:
                insights.append(f"{col} has only one unique value")
            elif df[col].nunique() > df.shape[0] * 0.8:
                insights.append(f"{col} has mostly unique values")

        return {
            "document_id": document_id,
            "analysis_type": "statistics",
            "summary": f"Statistical analysis of {len(stats)} columns",
            "statistics": stats,
            "insights": insights,
            "recommendations": recommendations,
        }

    def _analyze_text_summary(self, text: str, document_id: str) -> Dict[str, Any]:
        """Analyze text content for summary"""
        insights = []
        recommendations = []

        # Basic text statistics
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(text.split("\n"))

        # Readability metrics
        sentences = re.split(r"[.!?]+", text)
        sentence_count = len([s for s in sentences if s.strip()])

        avg_words_per_sentence = word_count / max(sentence_count, 1)
        avg_chars_per_word = char_count / max(word_count, 1)

        # Common words analysis
        words = re.findall(r"\b\w+\b", text.lower())
        word_freq = Counter(words)
        common_words = [word for word, _ in word_freq.most_common(10) if len(word) > 3]

        insights.append(
            f"Document contains {word_count} words in {sentence_count} sentences"
        )
        insights.append(f"Average {avg_words_per_sentence:.1f} words per sentence")

        if common_words:
            insights.append(f"Common themes: {', '.join(common_words[:5])}")

        # Content type detection
        if any(
            keyword in text.lower()
            for keyword in ["function", "class", "import", "def "]
        ):
            insights.append("Appears to contain code or technical content")
            recommendations.append("Consider syntax highlighting for code sections")

        if any(keyword in text.lower() for keyword in ["http", "www.", ".com", ".org"]):
            insights.append("Contains web links or URLs")
            recommendations.append("Validate and update links if needed")

        return {
            "document_id": document_id,
            "analysis_type": "summary",
            "summary": f"Text document with {word_count} words, {sentence_count} sentences",
            "statistics": {
                "word_count": word_count,
                "character_count": char_count,
                "line_count": line_count,
                "sentence_count": sentence_count,
                "avg_words_per_sentence": round(avg_words_per_sentence, 2),
                "avg_chars_per_word": round(avg_chars_per_word, 2),
            },
            "insights": insights,
            "recommendations": recommendations,
        }

    def _analyze_text_statistics(self, text: str, document_id: str) -> Dict[str, Any]:
        """Generate statistical analysis for text"""
        stats = {}
        insights = []

        # Word frequency analysis
        words = re.findall(r"\b\w+\b", text.lower())
        word_freq = Counter(words)

        # Remove common stop words for better analysis
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "shall",
        }
        filtered_words = [
            word for word in words if word not in stop_words and len(word) > 2
        ]

        filtered_freq = Counter(filtered_words)

        stats["word_frequency"] = dict(word_freq.most_common(20))
        stats["filtered_word_frequency"] = dict(filtered_freq.most_common(20))

        # Sentence analysis
        sentences = re.split(r"[.!?]+", text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]

        if sentence_lengths:
            stats["sentence_length_stats"] = {
                "mean": float(np.mean(sentence_lengths)),
                "median": float(np.median(sentence_lengths)),
                "min": int(np.min(sentence_lengths)),
                "max": int(np.max(sentence_lengths)),
                "std": float(np.std(sentence_lengths)),
            }

        # Character analysis
        char_freq = Counter(text.lower())
        stats["character_frequency"] = dict(char_freq.most_common(10))

        # Insights
        if len(words) > 1000:
            insights.append("Long document - consider breaking into sections")
        if len(set(words)) / len(words) > 0.8:
            insights.append("High lexical diversity - rich vocabulary")
        if np.mean(sentence_lengths) > 25:
            insights.append("Long sentences detected - may affect readability")

        return {
            "document_id": document_id,
            "analysis_type": "statistics",
            "summary": f"Detailed text statistics for {len(words)} words",
            "statistics": stats,
            "insights": insights,
            "recommendations": [],
        }
