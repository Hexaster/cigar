"""TabPFN regressor factory.

The API token is read from TABPFN_API_TOKEN so it never lives in the repo.
"""

from __future__ import annotations

import os


def make_tabpfn_regressor(random_state: int = 42):
    """Build a TabPFN regressor, reading any API token from the environment."""
    try:
        import tabpfn_client
        from tabpfn_client import TabPFNRegressor
    except ModuleNotFoundError:
        try:
            from tabpfn import TabPFNRegressor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "TabPFN is required for the paper experiments. Install "
                "`tabpfn-client` or `tabpfn`, and set TABPFN_API_TOKEN if "
                "your client requires authentication."
            ) from exc
        return TabPFNRegressor(random_state=random_state)

    token = os.environ.get("TABPFN_API_TOKEN")
    if token and hasattr(tabpfn_client, "set_access_token"):
        tabpfn_client.set_access_token(token)
    return TabPFNRegressor(random_state=random_state)
