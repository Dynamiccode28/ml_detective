import pandas as pd
import pytest

from ml_detective.ingestion.loader import load_csv
from ml_detective.utils.exceptions import EmptyDatasetError, UnsupportedFileTypeError

SAMPLE_CSV_PATH = "data/sample/employee_churn_synthetic.csv"


def test_load_csv_returns_a_dataframe_with_expected_shape():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.shape[0] == 200
    assert "churned" in dataframe.columns


def test_load_csv_rejects_non_csv_file_extension(tmp_path):
    fake_excel_file = tmp_path / "data.xlsx"
    fake_excel_file.write_text("not really a csv")

    with pytest.raises(UnsupportedFileTypeError):
        load_csv(fake_excel_file)


def test_load_csv_rejects_a_file_with_only_headers(tmp_path):
    header_only_file = tmp_path / "empty.csv"
    header_only_file.write_text("col_a,col_b,col_c\n")

    with pytest.raises(EmptyDatasetError):
        load_csv(header_only_file)


def test_load_csv_rejects_a_completely_empty_file(tmp_path):
    truly_empty_file = tmp_path / "blank.csv"
    truly_empty_file.write_text("")

    with pytest.raises(EmptyDatasetError):
        load_csv(truly_empty_file)