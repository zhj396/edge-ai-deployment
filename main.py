"""YOLOv8s ONNX Runtime Toolchain — CLI entry point.

Subcommands
-----------
* ``export``      — PyTorch .pt -> ONNX FP32
* ``inspect``     — print ONNX model metadata (sha256, opset, shapes)
* ``quantize``    — ONNX FP32 -> ONNX INT8 (static PTQ)
* ``infer``       — batch inference + visualization

All commands share the same top-level ``--log-level`` flag.
"""
import argparse
import logging
import sys

from cli import export, infer, inspect, quantize
from src import __version__ as PACKAGE_VERSION
from utils import get_logger, setup_logging, suppress_third_party_logs

logger = get_logger(__name__)


COMMANDS = {
    "export": export,
    "inspect": inspect,
    "quantize": quantize,
    "infer": infer,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="yolov8s-ort",
        description="YOLOv8s 12-Class ONNX Runtime Toolchain",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python main.py export --model models/yolov8s.pt "
               "--output models/yolov8s_fp32.onnx --opset 17",
    )
    parser.add_argument(
        "--log-level", type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO", help="Set logging level",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, mod in COMMANDS.items():
        if hasattr(mod, "add_parser"):
            mod.add_parser(sub)
        else:
            logger.warning("Module %s is missing add_parser", name)
    return parser.parse_args()


_LEVELS = {
    "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
    "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL,
}


def main() -> int:
    args = parse_args()
    setup_logging(level=_LEVELS[args.log_level])
    suppress_third_party_logs()

    logger.info("=" * 50)
    logger.info("YOLOv8s 12-Class ONNX Runtime Toolchain v%s", PACKAGE_VERSION)
    logger.info("=" * 50)

    if args.command not in COMMANDS:
        logger.error("Unknown command: %s", args.command)
        return 1

    try:
        rc = COMMANDS[args.command].run(args)
        # Subcommands may return an int exit code; None means success (run()
        # implementations that don't return a value exit 0).
        return int(rc) if isinstance(rc, int) else 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Command '%s' failed", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
