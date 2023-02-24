#!/usr/bin/env python

"""
Q-score
"""


def main():
    import argparse

    import qscore

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Q-score {qscore.__version__}",
    )

    import qscore.run_qscore

    modules = {
        "calculate": qscore.run_qscore,
    }

    subparsers = parser.add_subparsers(title="Choose a module",)
    subparsers.required = "True"

    for key in modules:
        module_parser = subparsers.add_parser(
            key,
            description=modules[key].__doc__,
            formatter_class=argparse.RawTextHelpFormatter,
        )
        modules[key].add_args(module_parser)
        module_parser.set_defaults(func=modules[key].main)

    try:
        args = parser.parse_args()
        args.func(args)
    except TypeError:
        parser.print_help()


if __name__ == "__main__":
    main()

