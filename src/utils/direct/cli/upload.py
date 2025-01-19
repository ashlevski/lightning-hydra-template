# coding=utf-8
# Copyright (c) DIRECT Contributors
import argparse
import os

from src.utils.direct.cli.utils import is_file
from src.utils.direct.utils.io import upload_to_s3


def upload_from_argparse(args: argparse.Namespace):  # pragma: no cover
    upload_to_s3(
        filename=args.data,
        to_filename=args.upload_path,
        endpoint_url=args.aws_endpoint_url,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        bucket=args.aws_bucket_name,
        verbose=not args.silent,
    )


class BaseArgs(argparse.ArgumentParser):  # pragma: no cover
    """Defines global default arguments."""

    def __init__(self, epilog=None, add_help=True, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """
        super().__init__(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=add_help)

        self.add_argument("--silent", help="Do not show progress", action="store_true")
        self.set_defaults(**overrides)


def register_parser(parser: argparse._SubParsersAction):  # pragma: no cover
    """Register upload commands to a root parser."""

    epilog = """
        """
    common_parser = BaseArgs(add_help=False)
    upload_parser = parser.add_parser(
        "upload-to-bucket",
        help="Upload data to S3 bucket.",
        parents=[common_parser],
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    aws_access_key_id = os.environ.get("DIRECT_AWS_ACCESS_KEY_ID", "direct")
    aws_secret_access_key = os.environ.get("DIRECT_AWS_SECRET_ACCESS_KEY", None)
    upload_parser.add_argument(
        "--aws_endpoint_url",
        type=str,
        help="S3 endpoint url.",
        default="https://s3.aiforoncology.nl",
    )
    upload_parser.add_argument("data", type=is_file, help="File to upload.")
    upload_parser.add_argument("upload_path", type=str, help="Path where to upload.")

    upload_parser.add_argument(
        "--aws-access-key-id",
        type=str,
        help="S3 access key id, (default='direct'). "
        "Can also be set with environmental variable DIRECT_AWS_ACCESS_KEY_ID",
        default=aws_access_key_id,
        required=aws_access_key_id is None,
    )
    upload_parser.add_argument(
        "--aws-secret-access-key",
        type=str,
        default=aws_secret_access_key,
        help="S3 secret access key. Can also be set with environmental variable DIRECT_AWS_SECRET_ACCESS_KEY",
        required=aws_secret_access_key is None,
    )

    upload_parser.add_argument(
        "--aws-bucket-name",
        type=str,
        default="direct-project",
        help="S3 bucket name",
    )

    upload_parser.set_defaults(subcommand=upload_from_argparse)
