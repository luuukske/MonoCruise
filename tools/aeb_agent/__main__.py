"""Entry point so `python -m tools.aeb_agent` reaches the CLI."""

import sys

from tools.aeb_agent.cli import main

if __name__ == "__main__":
    sys.exit(main())
