"""Reporting a failed execution without throwing away why it failed.

`describe` used to keep the first 400 characters of boa's message.  A frame
carries its arguments, and a six-leg route's arguments are longer than that, so
every multi-leg failure reported the call and cut off before the cause: the CLI
printed `BoaError:` followed by a truncated tuple.  Diagnosing one revert took
three runs and a monkeypatch to recover a line boa had already produced.
"""

from __future__ import annotations

from erouter.dev.executor import DESCRIBE_LIMIT, describe


class Boa(Exception):
    """Stands in for a `BoaError`, whose `str` is a whole call trace."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text


#: The shape that defeated the head slice: one very long outer frame carrying
#: the arguments and its reason, then the inner frame that actually refused.
OUTER = ("[E] [471710] RouteExecutor.execute_route:433(legs = ["
         + ", ".join(f'("0x{i:040x}", 0, 1, 0, 2, 0, 1, 280)' for i in range(6))
         + "], amount_in = 389626955441872512255, dst_slot = 1, min_out = 0)"
         " <leg reverted>")
INNER = "    [E] [2794] Unknown contract 0x5c4952751CF5C9D4eA3ad84F3407C56Ba2342F13.0x84a88ad5"
TRACE = "=" * 72 + "\n" + OUTER + "\n    [1031] fine.0x70a08231\n" + INNER + "\n" + "=" * 72


def test_the_innermost_frame_survives():
    # The whole point: the address and selector that refused are what a reader
    # needs, and they sit at the far end of a message far over budget.
    got = describe(Boa(TRACE))
    assert "0x5c4952751CF5C9D4eA3ad84F3407C56Ba2342F13.0x84a88ad5" in got
    assert len(TRACE) > DESCRIBE_LIMIT


def test_the_outer_reason_survives():
    # `<leg reverted>` closes the outer frame, past its own arguments.
    assert "<leg reverted>" in describe(Boa(TRACE))


def test_a_short_message_is_untouched():
    assert describe(Boa("ValueError: nope")) == "Boa: ValueError: nope"


def test_a_long_trace_with_no_error_frames_still_fits():
    got = describe(Boa("x" * 5_000))
    assert len(got) <= DESCRIBE_LIMIT


def test_an_exception_that_cannot_render_still_reports():
    # The original reason for the try/except: boa's formatter raises on a frame
    # with no method id, and a reporter that cannot report is worse than a
    # short message.
    class Unrenderable(Exception):
        def __str__(self) -> str:
            raise RuntimeError("no method id")

    assert "could not render" in describe(Unrenderable())
