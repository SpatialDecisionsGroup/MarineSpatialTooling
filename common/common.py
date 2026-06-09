"""Common helpers for dataset preparation scripts."""

import re
from datetime import timedelta

import pandas as pd


def normalise_label(value):
	"""Normalise labels for folder and site matching."""
	value = value.strip().lower()
	value = value.replace("_", " ")
	value = value.replace("-", " ")
	value = re.sub(r"\s+", " ", value)
	return value


def clean_column_names(frame):
	"""Strip whitespace and normalise unnamed columns."""
	renamed = []
	for column in frame.columns:
		column_name = str(column).strip()
		if column_name == "" or column_name.startswith("Unnamed:"):
			column_name = "Date"
		renamed.append(column_name)

	frame = frame.copy()
	frame.columns = renamed
	return frame


def coverage_class(value):
	"""Map coverage percent values into low, medium, or high classes."""
	numeric_value = pd.to_numeric(value, errors="coerce")
	if pd.isna(numeric_value):
		return ""
	if numeric_value < 30:
		return "low"
	if numeric_value < 60:
		return "medium"
	return "high"


def parse_date_object(value):
	"""Parse a date-like value into a Timestamp when possible."""
	text = str(value).strip()
	if not text:
		return None
	if re.fullmatch(r"\d{8}", text):
		parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
	elif "T" in text:
		parsed = pd.to_datetime(text, errors="coerce")
	else:
		parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
	if pd.isna(parsed):
		return None
	return parsed


def parse_date_value(value):
	"""Convert a date-like value into YYYYMMDD text."""
	parsed = parse_date_object(value)
	if parsed is None:
		return ""
	return parsed.strftime("%Y%m%d")


def format_date_window(value, window_days):
	"""Return start and end dates around a date value."""
	parsed = parse_date_object(value)
	if parsed is None:
		return None, None
	start = (parsed - timedelta(days=window_days)).strftime("%Y-%m-%d")
	end = (parsed + timedelta(days=window_days)).strftime("%Y-%m-%d")
	return start, end


def combine_frames(frames):
	"""Concatenate dataset-specific frames and keep Station after Location where present."""
	combined_frames = [frame.copy() for _, frame in frames if not frame.empty]
	if not combined_frames:
		return pd.DataFrame()

	combined = pd.concat(combined_frames, ignore_index=True, sort=False)
	if "Station" in combined.columns:
		columns = list(combined.columns)
		columns.remove("Station")
		location_index = columns.index("Location")
		columns.insert(location_index + 1, "Station")
		combined = combined[columns]

	return combined