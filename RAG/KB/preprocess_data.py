import csv
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CSV_FILES = {
    'KPIs': 'RAG/Content/RawContentFiles/KPIs.csv',
    'Objectives': 'RAG/Content/RawContentFiles/Objectives.csv',
    'BSC_families': 'RAG/Content/RawContentFiles/BSC_families.csv',
    'BSC_subfamilies': 'RAG/Content/RawContentFiles/BSC_subfamilies.csv',
    'Criteria': 'RAG/Content/RawContentFiles/Criteria.csv'
}


def preprocess_data(output_dir: str = "ProcessedFiles") -> Dict[str, str]:
    """
       Preprocesses raw CSV files by cleaning them and saving cleaned versions to the specified output directory.

       Args:
           output_dir (str): Path to the directory where cleaned CSV files will be saved.
                            Defaults to "ProcessedFiles".

       Returns:
           Dict[str, str]: A dictionary mapping each CSV type (e.g. 'KPIs') to its cleaned file path.
       """


    logger.info(f"Preprocessing data files to: {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    file_paths = list(CSV_FILES.values())
    cleaned_paths = clean_all_csv_files(file_paths, output_dir)

    processed_files = {}
    for file_type, file_path in CSV_FILES.items():
        if file_path in cleaned_paths and cleaned_paths[file_path]:
            processed_files[file_type] = cleaned_paths[file_path]
            logger.info(f"Processed {file_type}: {cleaned_paths[file_path]}")
        else:
            logger.warning(f"Failed to process {file_type}")

    return processed_files


def clean_csv_file(input_path: str, output_path: Optional[str] = None) -> str:

    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}.clean.csv"

    logger.info(f"Cleaning CSV file: {input_path}")

    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            content = file.readlines()

        header_line = content[0]
        headers = [h.strip() for h in header_line.split(',')]

        logger.info(f"Headers: {headers}")
        logger.info(f"Total lines in the original file: {len(content)}")

        processed_rows = []

        processed_rows.append(headers)

        buffer = []
        in_quoted_field = False

        for i, line in enumerate(content[1:], 1):
            if not line.strip():
                continue  # Skip empty lines

            line_chars = line.strip()

            for char in line_chars:
                if char == '"':
                    in_quoted_field = not in_quoted_field

            buffer.append(line.strip())

            if not in_quoted_field:
                full_line = " ".join(buffer)
                buffer = []  # Reset buffer

                parsed_row = parse_csv_row(full_line, headers)
                if parsed_row:
                    processed_rows.append(parsed_row)

        if buffer:
            full_line = " ".join(buffer)
            parsed_row = parse_csv_row(full_line, headers)
            if parsed_row:
                processed_rows.append(parsed_row)

        logger.info(f"Processed {len(processed_rows) - 1} data rows")

        write_clean_csv(processed_rows, output_path)

        logger.info(f"Cleaned CSV saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error cleaning CSV file: {str(e)}")
        raise e


def parse_csv_row(line: str, headers: List[str]) -> List[str]:

    fields = []
    current_field = ""
    in_quotes = False

    for char in line:
        if char == '"':
            # Toggle quote state
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            # End of field
            fields.append(current_field.strip())
            current_field = ""
        else:
            current_field += char

    # Add the last field
    if current_field:
        fields.append(current_field.strip())

    # Make sure we have the right number of fields
    if len(fields) < len(headers):
        # Pad with empty strings
        fields.extend([""] * (len(headers) - len(fields)))
    elif len(fields) > len(headers):
        # Combine extra fields into the last one
        extra = ", ".join(fields[len(headers):])
        fields = fields[:len(headers) - 1] + [fields[len(headers) - 1] + ", " + extra]

    # Clean JSON fields
    for i, field in enumerate(fields):
        fields[i] = clean_field(field)

    return fields


def clean_field(field: str) -> str:

    # Remove leading/trailing whitespace
    field = field.strip()

    # Handle quoted fields - remove outer quotes if present
    if field.startswith('"') and field.endswith('"'):
        field = field[1:-1]

    # Handle escaped quotes inside the field
    field = field.replace('""', '"')

    # Handle JSON arrays - try to parse and standardize
    if field.startswith('[') and field.endswith(']'):
        try:
            # Parse as JSON
            parsed = json.loads(field)
            if isinstance(parsed, list):
                # Convert back to a standardized JSON string
                return json.dumps(parsed)
        except:
            pass  # If parsing fails, keep as is

    # Replace problematic characters
    field = field.replace('\n', ' ').replace('\r', ' ')

    return field


def write_clean_csv(rows: List[Any], output_path: str) -> None:

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(
            csvfile,
            quoting=csv.QUOTE_MINIMAL,  # Only quote fields that need it
            escapechar='\\',  # Use backslash as escape character
            doublequote=False  # Don't double quotes for escaping
        )

        # Write each row
        for row in rows:
            writer.writerow(row)


def clean_all_csv_files(file_paths: List[str], output_dir: Optional[str] = None) -> Dict[str, str]:

    cleaned_paths = {}

    for file_path in file_paths:
        if output_dir:
            base_name = os.path.basename(file_path)
            output_path = os.path.join(output_dir, f"clean_{base_name}")
        else:
            base_name = os.path.splitext(file_path)[0]
            output_path = f"{base_name}.clean.csv"

        try:
            cleaned_path = clean_csv_file(file_path, output_path)
            cleaned_paths[file_path] = cleaned_path
        except Exception as e:
            logger.error(f"Failed to clean {file_path}: {str(e)}")
            cleaned_paths[file_path] = None

    return cleaned_paths