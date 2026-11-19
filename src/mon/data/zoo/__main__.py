"""Dataset builders: ``python -m mon.data.zoo testkit build [--out DIR] [--overwrite]``."""

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mon.data.zoo", description="Build derived datasets.")
    sub = parser.add_subparsers(dest="dataset", required=True)
    tk = sub.add_parser("testkit", help="the fast-test subset (mon.data.zoo.testkit)")
    tk_sub = tk.add_subparsers(dest="cmd", required=True)
    b = tk_sub.add_parser("build", help="build it from the source datasets")
    b.add_argument("--out", help="output directory (default: $DATASETS_DIR/testkit)")
    b.add_argument("--overwrite", action="store_true", help="replace existing files (none are deleted)")
    args = parser.parse_args(argv)

    from mon.core.console import setup_logging
    from mon.data.zoo.testkit import build

    setup_logging()
    print(build(args.out, overwrite=args.overwrite))
    return 0


if __name__ == "__main__":
    sys.exit(main())
