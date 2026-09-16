# ============================================================================
# Example: ETLEngine - data cleaning pipeline with audit report
# Run:  python examples/demo_etl.py
# ============================================================================

"""Cleans a messy DataFrame and prints the before/after + audit report."""

import pandas as pd

from agentx.etl import DataCleaner

if __name__ == "__main__":
    messy = pd.DataFrame(
        {
            "name": ["Alice", "Alice", " bob ", "Carol", None, "David"],
            "age": [30, 30, None, 28, 45, 300],
            "city": ["beijing ", "beijing ", "shanghai", "guangzhou", "shenzhen", "beijing"],
            "signup": ["2026-01-05", "2026-01-05", "bad-date", "2026-03-12", "2026-04-01", "2026-05-20"],
        }
    )
    print("BEFORE:")
    print(messy)

    cleaner = DataCleaner(messy).clean(cap_outliers=True)
    cleaned = cleaner.to_dataframe()

    print("\nAFTER:")
    print(cleaned)
    print("\nREPORT:")
    report = cleaner.report()
    print(
        f"rows: {report.rows_before} -> {report.rows_after} "
        f"(duplicates removed: {report.duplicates_removed})"
    )
    print(f"missing filled: {report.missing_filled}")
    print(f"outliers capped: {report.outliers_capped}")
    print(f"dtypes fixed: {report.dtypes_fixed}")

    # --- OCR (only if pytesseract + tesseract are installed) ----------------
    from agentx.etl import OcrEngine

    engine = OcrEngine(lang="eng")
    print(f"\nOCR available: {engine.is_available()}")
