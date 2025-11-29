"""
Allow running BBERT as a module: python -m bbert

This is equivalent to running the `bbert` command.
"""

from bbert.cli.main import main

if __name__ == "__main__":
    main()
