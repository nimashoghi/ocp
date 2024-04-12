import argparse
import contextlib

from ....common.flags import flags
from ....common.utils import build_config, new_trainer_context, setup_logging


@contextlib.contextmanager
def test_context(config_path: str):
    setup_logging()

    parser: argparse.ArgumentParser = flags.get_parser()
    args: argparse.Namespace
    args, override_args = parser.parse_known_args(
        [
            "--config",
            config_path,
            "--mode",
            "train",
            "--debug",
            "--amp",
        ]
    )

    config = build_config(args, override_args)
    with new_trainer_context(args=args, config=config) as ctx:
        ctx.task.setup(ctx.trainer)
        yield ctx
