# coding=utf-8
# Copyright (c) DIRECT Contributors
"""DIRECT Command-line interface.

This is the file which builds the main parser.
"""

import argparse


def main():
    """Console script for direct."""
    # From https://stackoverflow.com/questions/17073688/how-to-use-argparse-subparsers-correctly
    root_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    root_subparsers = root_parser.add_subparsers(help="Direct CLI utils to run.")
    root_subparsers.required = True
    root_subparsers.dest = "subcommand"

    # Prevent circular imports
    from src.utils.direct.cli.predict import register_parser as register_predict_subcommand
    from src.utils.direct.cli.train import register_parser as register_train_subcommand
    from src.utils.direct.cli.upload import register_parser as register_upload_subcommand

    # Training images related commands.
    register_train_subcommand(root_subparsers)
    # Inference images related commands.
    register_predict_subcommand(root_subparsers)
    # Data related comments.
    register_upload_subcommand(root_subparsers)

    args = root_parser.parse_args()
    args.subcommand(args)


if __name__ == "__main__":
    main()
