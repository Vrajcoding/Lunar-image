"""
Evaluation package for benchmark experiments, synthetic ground truth datasets, and metric reporting.
"""
from app.evaluation.dataset import SyntheticDatasetGenerator, load_image_pair
from app.evaluation.benchmark import BenchmarkRunner, run_full_benchmark
from app.evaluation.report import generate_benchmark_markdown_report

__all__ = [
    "SyntheticDatasetGenerator",
    "load_image_pair",
    "BenchmarkRunner",
    "run_full_benchmark",
    "generate_benchmark_markdown_report",
]
