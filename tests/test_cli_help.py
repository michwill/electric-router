"""`--help` must render, for every subcommand.

argparse runs each help string through `%` interpolation, so a literal
percent is a crash rather than a typo: `"5% of the size"` reads as a space-
flagged octal conversion and `erouter route --help` died with "%o format: an
integer is required, not dict".  Nothing else in the suite calls `format_help`,
so the CLI's own front door was the one path with no coverage.

Rendering every parser is the whole test; the bad string cannot survive it.
"""

import pytest

from erouter.dev.cli import build_parser


def _subparsers(parser):
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if choices and hasattr(choices, "items"):
            yield from choices.items()


PARSERS = [("erouter", build_parser())]
PARSERS += [(name, sub) for name, sub in _subparsers(build_parser())]


@pytest.mark.parametrize("name,parser", PARSERS, ids=[n for n, _ in PARSERS])
def test_help_renders(name, parser):
    assert parser.format_help()


def test_every_subcommand_is_covered():
    # A guard on the guard: if the parametrisation silently found no
    # subcommands, every case above would pass while testing nothing.
    assert len(PARSERS) > 5
