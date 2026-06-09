"""
Setup credentials for dataset creation and download.

Sets up credentials: Planet API key in a JSON file and Earth Engine via OAuth.
"""

import json
from pathlib import Path

import ee

from superres.constants import CREDENTIALS_CONFIG_FILENAME
from superres.config import build_credentials_parser


def _load_json_file(file_path):
    try:
        with open(file_path) as file_handle:
            return json.load(file_handle)
    except (json.JSONDecodeError, OSError):
        return None


def load_existing_credentials(output_dir):
    """Load existing credentials from credentials.json if it exists."""
    combined_file = output_dir / CREDENTIALS_CONFIG_FILENAME
    if combined_file.exists():
        credentials = _load_json_file(combined_file)
        if isinstance(credentials, dict):
            return credentials
    return {}


def prompt_gee_project(current_value):
    """Prompt for GCP project ID while allowing Enter to keep the current value."""
    suffix = "[set]" if current_value else "[empty]"
    response = input(f"Google Earth Engine project ID {suffix}: ").strip()

    if response:
        return response

    if current_value:
        return current_value

    print("Warning: leaving GCP project ID empty. You may need to set it later.")
    return None


def prompt_planet_key(current_value):
    """Prompt for a Planet API key while allowing Enter to keep the current value."""
    suffix = "[set]" if current_value else "[empty]"
    response = input(f"Planet Labs API key {suffix}: ").strip()

    if response:
        return response

    if current_value:
        return current_value

    print("Warning: leaving Planet Labs API key empty.")
    return None


def save_credentials(output_file, planet_key, gee_project):
    """Write the credentials config file."""
    credentials = {
        "api_key": planet_key or "",
        "gee_project": gee_project or "",
    }

    with open(output_file, "w") as file_handle:
        json.dump(credentials, file_handle, indent=2)

    output_file.chmod(0o600)
    print(f"Credentials saved to {output_file}")


args = build_credentials_parser().parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

existing_credentials = load_existing_credentials(output_dir)

print("Configure dataset credentials")
print(f"Output file: {output_dir / CREDENTIALS_CONFIG_FILENAME}")

if args.planet_key is not None:
    planet_key = args.planet_key
else:
    planet_key = prompt_planet_key(existing_credentials.get("api_key"))

gee_project = prompt_gee_project(existing_credentials.get("gee_project"))

if not planet_key:
    print("Warning: Planet API key is empty. PlanetScope downloads will not work until it is set.")

if not gee_project:
    print("Warning: GCP project ID is empty. Sentinel-2 downloads will not work until it is set.")

print("\nSetting up Earth Engine authentication...")
try:
    ee.Authenticate()
    print("Earth Engine authenticated successfully.")
except Exception as e:
    print(f"Warning: Earth Engine authentication failed: {e}")
    print("You can re-run setup_credentials.py later to complete authentication.")

save_credentials(output_dir / CREDENTIALS_CONFIG_FILENAME, planet_key, gee_project)
