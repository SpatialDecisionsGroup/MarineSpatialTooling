"""Credential helpers for dataset scripts."""

import json
from pathlib import Path


def load_credentials(path=None):
	repo_root = Path(__file__).resolve().parents[1]
	default = repo_root / "common" / "credentials.json"
	file_path = Path(path) if path else default
	if not file_path.exists():
		raise FileNotFoundError(f"Credentials file not found: {file_path}")
	with file_path.open("r", encoding="utf-8") as file_handle:
		return json.load(file_handle)
