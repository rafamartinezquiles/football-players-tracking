"""
secrets.py

This module is responsible for loading API keys and tokens needed by the project.

We do NOT hardcode secrets in code.

Expected environment variables:
- HF_TOKEN:           HuggingFace access token
- ROBOFLOW_API_KEY:   Roboflow API key

You can set them, for example:
- In your shell:
    export HF_TOKEN="..."
    export ROBOFLOW_API_KEY="..."

- In Colab:
    import os
    os.environ["HF_TOKEN"] = "<your HF token>"
    os.environ["ROBOFLOW_API_KEY"] = "<your key>"

This module provides a single helper `load_api_keys` which returns both.
"""

import os
from typing import Tuple


def load_api_keys() -> Tuple[str, str]:
    """
    Load HuggingFace and Roboflow API keys from environment variables.

    Returns:
        (hf_token, roboflow_api_key)

    Raises:
        ValueError if any of the variables is missing.
    """
    hf_token = os.environ.get("HF_TOKEN")
    roboflow_api_key = os.environ.get("ROBOFLOW_API_KEY")

    if hf_token is None:
        raise ValueError(
            "HF_TOKEN environment variable is not set. "
            "Please export HF_TOKEN before running the project."
        )

    if roboflow_api_key is None:
        raise ValueError(
            "ROBOFLOW_API_KEY environment variable is not set. "
            "Please export ROBOFLOW_API_KEY before running the project."
        )

    return hf_token, roboflow_api_key
