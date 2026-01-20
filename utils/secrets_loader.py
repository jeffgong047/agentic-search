"""
Secrets Loader - Load API keys from local files into environment variables.

Usage:
    from utils.secrets_loader import load_secrets
    load_secrets()  # Call at application startup
"""

import os
from pathlib import Path


def load_secrets(secrets_dir: str = None) -> dict:
    """
    Load secrets from .secrets/ directory into environment variables.

    Args:
        secrets_dir: Path to secrets directory. Defaults to .secrets/ in project root.

    Returns:
        Dict of loaded environment variable names and masked values.
    """
    if secrets_dir is None:
        # Find project root (where .secrets/ should be)
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / ".secrets").exists():
                secrets_dir = parent / ".secrets"
                break
        else:
            # Default to current working directory
            secrets_dir = Path.cwd() / ".secrets"
    else:
        secrets_dir = Path(secrets_dir)

    loaded = {}

    # Mapping of secret files to environment variables
    secret_mapping = {
        "anthropic_key": "ANTHROPIC_API_KEY",
        "openai_key": "OPENAI_API_KEY",
        "postgres_password": "POSTGRES_PASSWORD",
    }

    for filename, env_var in secret_mapping.items():
        secret_file = secrets_dir / filename
        if secret_file.exists():
            try:
                value = secret_file.read_text().strip()
                if value and not value.startswith("sk-YOUR") and not value.startswith("your-"):
                    os.environ[env_var] = value
                    # Mask the value for logging
                    masked = value[:8] + "..." + value[-4:] if len(value) > 15 else "***"
                    loaded[env_var] = masked
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")

    return loaded


def ensure_api_key(prefer_anthropic: bool = True) -> str:
    """
    Ensure at least one API key is available.

    Args:
        prefer_anthropic: If True, check Anthropic key first.

    Returns:
        The name of the available API key environment variable.

    Raises:
        ValueError: If no API key is found.
    """
    # Try to load from files first
    load_secrets()

    if prefer_anthropic:
        keys_to_check = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    else:
        keys_to_check = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]

    for key in keys_to_check:
        if os.environ.get(key):
            return key

    raise ValueError(
        "No API key found. Either:\n"
        "1. Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable\n"
        "2. Create .secrets/anthropic_key or .secrets/openai_key file"
    )


if __name__ == "__main__":
    # Test loading
    loaded = load_secrets()
    if loaded:
        print("Loaded secrets:")
        for key, masked in loaded.items():
            print(f"  {key}: {masked}")
    else:
        print("No secrets found. Create files in .secrets/ directory.")
        print("See .secrets/README.md for instructions.")
