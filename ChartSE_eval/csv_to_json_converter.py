#!/usr/bin/env python3
"""
CSV to JSON Converter Script

This script converts CSV files from a directory into a JSON format suitable for chart data extraction.
It handles various CSV formats including simple key-value pairs and time series data.

Usage:
    python csv_to_json_converter.py <input_directory> <output_file>

Input CSV formats supported:
1. Simple two-column format (e.g., Year, Value)
2. Multi-column format (e.g., Year, Cases, Deaths)
3. Date-based format (e.g., Date, Frequency)
4. Time series format (e.g., Day/Week, Cases, Deaths)

Output JSON format:
- Array of objects with "images" and "gts" fields
- "gts" contains title, source, x_title, y_title, and values
- "values" can be simple key-value pairs or nested objects for time series
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Union


def detect_csv_format(csv_file_path: str) -> str:
    """
    Detect the format of a CSV file by examining its structure.

    Args:
        csv_file_path: Path to the CSV file

    Returns:
        String indicating the detected format
    """
    with open(csv_file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)

        # Count columns
        num_cols = len(header)

        # Read a few rows to understand the data structure
        sample_rows = []
        for i, row in enumerate(reader):
            if i >= 3:  # Read first 3 data rows
                break
            sample_rows.append(row)

    if num_cols == 2:
        return "simple"
    elif num_cols == 3:
        return "time_series"
    else:
        return "multi_column"


def parse_simple_format(csv_file_path: str) -> Dict[str, Any]:
    """
    Parse CSV files with simple two-column format (e.g., Year, Value).

    Args:
        csv_file_path: Path to the CSV file

    Returns:
        Dictionary containing the parsed data
    """
    values = {}

    with open(csv_file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Get the first two columns
            cols = list(row.keys())
            if len(cols) >= 2:
                key = row[cols[0]].strip()
                value = row[cols[1]].strip()

                # Try to convert to number if possible
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string if conversion fails

                values[key] = value

    return {
        "title": "None",
        "source": "None",
        "x_title": "None",
        "y_title": "None",
        "values": values,
    }


def parse_time_series_format(csv_file_path: str) -> Dict[str, Any]:
    """
    Parse CSV files with time series format (e.g., Year, Cases, Deaths).

    Args:
        csv_file_path: Path to the CSV file

    Returns:
        Dictionary containing the parsed data
    """
    values = {}

    with open(csv_file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Get the first column as the time key
            time_cols = list(row.keys())
            time_key = row[time_cols[0]].strip()

            # Create nested dictionary for other columns
            time_data = {}
            for col in time_cols[1:]:
                value = row[col].strip()
                try:
                    if "." in value:
                        time_data[col] = float(value)
                    else:
                        time_data[col] = int(value)
                except ValueError:
                    time_data[col] = value

            values[time_key] = time_data

    return {
        "title": "None",
        "source": "None",
        "x_title": "None",
        "y_title": "None",
        "values": values,
    }


def parse_multi_column_format(csv_file_path: str) -> Dict[str, Any]:
    """
    Parse CSV files with multiple columns format.

    Args:
        csv_file_path: Path to the CSV file

    Returns:
        Dictionary containing the parsed data
    """
    values = {}

    with open(csv_file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Get the first column as the key
            cols = list(row.keys())
            key = row[cols[0]].strip()

            # Create nested dictionary for other columns
            key_data = {}
            for col in cols[1:]:
                value = row[col].strip()
                try:
                    if "." in value:
                        key_data[col] = float(value)
                    else:
                        key_data[col] = int(value)
                except ValueError:
                    key_data[col] = value

            values[key] = key_data

    return {
        "title": "None",
        "source": "None",
        "x_title": "None",
        "y_title": "None",
        "values": values,
    }


def convert_csv_to_json(input_directory: str, output_file: str) -> None:
    """
    Convert all CSV files in a directory to JSON format.

    Args:
        input_directory: Path to directory containing CSV files
        output_file: Path to output JSON file
    """
    input_path = Path(input_directory)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_directory}")

    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_directory}")

    # Find all CSV files
    csv_files = list(input_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_directory}")
        return

    print(f"Found {len(csv_files)} CSV files to process")

    # Process each CSV file
    json_data = []

    for csv_file in sorted(csv_files):
        print(f"Processing {csv_file.name}...")

        try:
            # Detect format and parse accordingly
            format_type = detect_csv_format(str(csv_file))

            if format_type == "simple":
                gts_data = parse_simple_format(str(csv_file))
            elif format_type == "time_series":
                gts_data = parse_time_series_format(str(csv_file))
            else:  # multi_column
                gts_data = parse_multi_column_format(str(csv_file))

            # Create the JSON entry
            entry = {
                "images": csv_file.stem + ".png",  # Assume corresponding image file
                "gts": gts_data,
            }

            json_data.append(entry)

        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
            continue

    # Write to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=2, ensure_ascii=False)

    print(f"Successfully converted {len(json_data)} files to {output_file}")


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description="Convert CSV files to JSON format for chart data extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python csv_to_json_converter.py /path/to/csv/files output.json
    python csv_to_json_converter.py ./data/CholeraEpicurves/ground_truth ./output/chart_data.json
        """,
    )

    parser.add_argument(
        "input_directory", help="Path to directory containing CSV files"
    )

    parser.add_argument("output_file", help="Path to output JSON file")

    args = parser.parse_args()

    try:
        convert_csv_to_json(args.input_directory, args.output_file)
        print("Conversion completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
