"""
exceptions.py

Defines our own family of error types, instead of using generic
built-in errors like ValueError everywhere.
"""


class MLDetectiveError(Exception):
    """The base error for this entire project."""


class ConfigurationError(MLDetectiveError):
    """Raised when a required setting is missing or invalid at startup."""


class IngestionError(MLDetectiveError):
    """Base error for anything that goes wrong while reading an input file."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when the uploaded file is not a type we support (e.g. not a CSV)."""


class EmptyDatasetError(IngestionError):
    """Raised when an uploaded file has no rows of actual data in it."""


class SchemaValidationError(MLDetectiveError):
    """Raised when a dataset's structure doesn't match what's expected."""


class AnalysisError(MLDetectiveError):
    """Raised when a statistical or analytical calculation fails unexpectedly."""