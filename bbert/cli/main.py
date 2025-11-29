#!/usr/bin/env python
"""
BBERT Command Line Interface

Main entry point for BBERT CLI commands.
"""

import sys
import argparse
from bbert._version import __version__


def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        prog='bbert',
        description='BBERT - BERT for Bacterial DNA Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bbert infer examples/data/sample.fasta --output-dir results
  bbert download
  bbert --version

For command-specific help:
  bbert infer --help
  bbert download --help
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'BBERT {__version__}'
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title='commands',
        description='Available BBERT commands',
        dest='command',
        help='Command to run'
    )

    # Import command modules
    from bbert.cli.commands import infer, download

    # Register infer command
    infer_parser = subparsers.add_parser(
        'infer',
        help='Run BBERT inference on DNA sequences',
        description='Run BBERT inference on FASTA/FASTQ files'
    )
    infer.add_arguments(infer_parser)
    infer_parser.set_defaults(func=infer.main)

    # Register download command
    download_parser = subparsers.add_parser(
        'download',
        help='Download BBERT models from HuggingFace',
        description='Download required model files from HuggingFace Hub'
    )
    download.add_arguments(download_parser)
    download_parser.set_defaults(func=download.main)

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, show help
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)

    # Run the command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
