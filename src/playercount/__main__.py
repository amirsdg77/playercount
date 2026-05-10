"""Allow ``python -m playercount`` to dispatch to the Typer CLI."""

from __future__ import annotations

from playercount.cli import app

if __name__ == "__main__":
    app()
