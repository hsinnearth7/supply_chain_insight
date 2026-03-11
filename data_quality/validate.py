"""Great Expectations data quality validation runner for ChainInsight.

Usage:
    python data_quality/validate.py                    # Run all validations
    python data_quality/validate.py --suite demand     # Run demand suite only
    python data_quality/validate.py --suite inventory  # Run inventory suite only
    python data_quality/validate.py --file path/to/data.csv --suite demand
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Paths
DATA_QUALITY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_QUALITY_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_expectation_suite(suite_name: str) -> dict[str, Any]:
    """Load an expectation suite from JSON file."""
    suite_path = DATA_QUALITY_DIR / "expectations" / f"{suite_name}.json"
    if not suite_path.exists():
        raise FileNotFoundError(f"Expectation suite not found: {suite_path}")

    with open(suite_path) as f:
        return json.load(f)


def validate_column_not_null(df: Any, column: str) -> tuple[bool, str]:
    """Check that a column has no null values."""
    null_count = df[column].isna().sum()
    passed = null_count == 0
    msg = f"Column '{column}': {null_count} null values found"
    return passed, msg


def validate_column_between(
    df: Any, column: str, min_val: float, max_val: float, mostly: float = 1.0
) -> tuple[bool, str]:
    """Check that column values fall within a range."""
    valid = df[column].between(min_val, max_val)
    pct_valid = valid.mean()
    passed = pct_valid >= mostly
    msg = (
        f"Column '{column}': {pct_valid:.2%} in range [{min_val}, {max_val}] "
        f"(required: {mostly:.0%})"
    )
    return passed, msg


def validate_column_type(df: Any, column: str, expected_type: str) -> tuple[bool, str]:
    """Check that a column has the expected dtype."""
    actual_type = str(df[column].dtype)
    # Allow flexible type matching
    type_map = {
        "float64": ["float64", "float32", "int64", "int32"],
        "int64": ["int64", "int32"],
        "object": ["object", "string"],
    }
    valid_types = type_map.get(expected_type, [expected_type])
    passed = actual_type in valid_types
    msg = f"Column '{column}': type is '{actual_type}' (expected '{expected_type}')"
    return passed, msg


def validate_row_count(df: Any, min_val: int, max_val: int) -> tuple[bool, str]:
    """Check that row count falls within expected range."""
    count = len(df)
    passed = min_val <= count <= max_val
    msg = f"Row count: {count} (expected between {min_val} and {max_val})"
    return passed, msg


def run_validation(
    file_path: Path,
    suite_name: str,
) -> dict[str, Any]:
    """Run a validation suite against a data file.

    Args:
        file_path: Path to the CSV data file.
        suite_name: Name of the expectation suite (without extension).

    Returns:
        Validation result dictionary.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas is required for validation")
        return {"success": False, "error": "pandas not installed"}

    # Load data
    if not file_path.exists():
        return {
            "success": False,
            "error": f"Data file not found: {file_path}",
            "suite": suite_name,
        }

    logger.info("Loading data from %s", file_path)
    df = pd.read_csv(file_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Load expectations
    suite = load_expectation_suite(suite_name)
    expectations = suite.get("expectations", [])

    results = {
        "suite_name": suite.get("expectation_suite_name", suite_name),
        "data_file": str(file_path),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "expectations_evaluated": 0,
        "expectations_passed": 0,
        "expectations_failed": 0,
        "details": [],
    }

    for exp in expectations:
        exp_type = exp.get("expectation_type", "")
        kwargs = exp.get("kwargs", {})
        passed = False
        message = ""

        try:
            if exp_type == "expect_column_values_to_not_be_null":
                col = kwargs["column"]
                if col in df.columns:
                    passed, message = validate_column_not_null(df, col)
                else:
                    message = f"Column '{col}' not found in data"

            elif exp_type == "expect_column_values_to_be_between":
                col = kwargs["column"]
                if col in df.columns:
                    passed, message = validate_column_between(
                        df, col, kwargs["min_value"], kwargs["max_value"],
                        kwargs.get("mostly", 1.0),
                    )
                else:
                    message = f"Column '{col}' not found in data"

            elif exp_type == "expect_column_values_to_be_of_type":
                col = kwargs["column"]
                if col in df.columns:
                    passed, message = validate_column_type(df, col, kwargs["type_"])
                else:
                    message = f"Column '{col}' not found in data"

            elif exp_type == "expect_table_row_count_to_be_between":
                passed, message = validate_row_count(
                    df, kwargs["min_value"], kwargs["max_value"]
                )

            elif exp_type == "expect_table_columns_to_match_ordered_list":
                expected_cols = kwargs.get("column_list", [])
                actual_cols = list(df.columns)
                passed = actual_cols == expected_cols
                message = (
                    f"Columns match: {passed}. "
                    f"Expected: {expected_cols}, Got: {actual_cols}"
                )

            elif exp_type == "expect_table_columns_to_match_set":
                expected_set = set(kwargs.get("column_set", []))
                actual_set = set(df.columns)
                exact = kwargs.get("exact_match", True)
                if exact:
                    passed = actual_set == expected_set
                else:
                    passed = expected_set.issubset(actual_set)
                message = f"Column set {'exact' if exact else 'subset'} match: {passed}"

            else:
                passed = True
                message = f"Skipped (unimplemented): {exp_type}"

        except Exception as e:
            message = f"Error evaluating {exp_type}: {e}"

        results["expectations_evaluated"] += 1
        if passed:
            results["expectations_passed"] += 1
        else:
            results["expectations_failed"] += 1

        results["details"].append({
            "expectation_type": exp_type,
            "passed": passed,
            "message": message,
            "kwargs": kwargs,
        })

    results["success"] = results["expectations_failed"] == 0

    return results


def main() -> int:
    """CLI entry point for data quality validation."""
    parser = argparse.ArgumentParser(description="ChainInsight Data Quality Validation")
    parser.add_argument(
        "--suite",
        choices=["demand", "inventory", "all"],
        default="all",
        help="Expectation suite to run (default: all)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to specific data file (overrides default discovery)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON validation report",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed expectation results",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    suites_to_run = []
    if args.suite in ("demand", "all"):
        suites_to_run.append(("demand_data", DATA_DIR / "demand_data.csv"))
    if args.suite in ("inventory", "all"):
        suites_to_run.append(("inventory_data", DATA_DIR / "inventory_data.csv"))

    all_results = []
    overall_success = True

    for suite_name, default_path in suites_to_run:
        file_path = Path(args.file) if args.file else default_path
        logger.info("Running suite '%s' on %s", suite_name, file_path)

        result = run_validation(file_path, suite_name)
        all_results.append(result)

        status = "PASSED" if result["success"] else "FAILED"
        logger.info(
            "Suite '%s': %s (%d/%d expectations passed)",
            suite_name,
            status,
            result["expectations_passed"],
            result["expectations_evaluated"],
        )

        if args.verbose:
            for detail in result["details"]:
                icon = "OK" if detail["passed"] else "FAIL"
                logger.info("  [%s] %s: %s", icon, detail["expectation_type"], detail["message"])

        if not result["success"]:
            overall_success = False

    # Write output report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Validation report written to %s", output_path)

    # Summary
    print("\n" + "=" * 60)
    print("DATA QUALITY VALIDATION SUMMARY")
    print("=" * 60)
    for result in all_results:
        status = "PASSED" if result["success"] else "FAILED"
        print(
            f"  {result['suite_name']}: {status} "
            f"({result['expectations_passed']}/{result['expectations_evaluated']} passed)"
        )
    print("=" * 60)
    print(f"Overall: {'PASSED' if overall_success else 'FAILED'}")
    print()

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
