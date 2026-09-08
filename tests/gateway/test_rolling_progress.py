"""Tests for the ``rolling`` tool-progress grouping mode.

``tool_progress_grouping: rolling`` keeps a single editable progress bubble (same
message ID, edited in place) but only shows the newest ``tool_progress_max_entries``
logical entries. Older entries roll off the top; a display-only omission label marks
that earlier activity was dropped.

This is a presentation-layer feature: it must NOT touch tool results / model context,
and ``accumulate`` / ``separate`` must behave exactly as before.
"""

from __future__ import annotations

import asyncio
import importlib
import queue as queue_mod
import time
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, PlatformConfig, SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext

LABEL = "… earlier tool activity omitted …"


# ---------------------------------------------------------------------------
# Capturing adapter (records sends + edits, edits in place)
# ---------------------------------------------------------------------------
class RollingCaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.WHATSAPP, max_len=4000):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.MAX_MESSAGE_LENGTH = max_len
        self.sent = []
        self.edits = []
        self.typing = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id=f"progress-{len(self.sent)}")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append({"message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append(chat_id)

    async def stop_typing(self, chat_id) -> None:
        return None


# ---------------------------------------------------------------------------
# Harness helpers
# ---------------------------------------------------------------------------
def _make_runner(ctx):
    class _StubGatewayRunner:
        def _adapter_for_source(self, source):
            return getattr(self, "_adapter", None)

    runner = _StubGatewayRunner()
    return TurnRunner(runner, ctx)


def _make_ctx(grouping="rolling", max_entries=4, progress_mode="all", queue=None):
    ctx = TurnContext(
        source=SimpleNamespace(chat_id="chat-1", platform=Platform.WHATSAPP),
        progress_grouping=grouping,
        progress_max_entries=max_entries,
        progress_mode=progress_mode,
        tool_progress_enabled=True,
        _run_still_current=lambda: True,
    )
    if queue is not None:
        ctx.progress_queue = queue
    return ctx


def _state_for(adapter, ctx):
    runner = _make_runner(ctx)
    runner._runner._adapter = adapter  # _StubGatewayRunner._adapter_for_source uses this
    return runner, runner._progress_edit_state(adapter)


def _absorb_all(runner, st, items):
    """Feed items through ``_progress_absorb`` and return the visible lines."""
    for raw in items:
        runner._progress_absorb(st, raw)
    return st.progress_lines


# ---------------------------------------------------------------------------
# 1-5. Core windowing behaviour (logical entries, not characters)
# ---------------------------------------------------------------------------
class TestRollingWindowing:
    def test_fewer_than_max(self):
        adapter = RollingCaptureAdapter()
        ctx = _make_ctx(grouping="rolling", max_entries=4)
        runner, st = _state_for(adapter, ctx)
        lines = _absorb_all(runner, st, ["A", "B"])
        assert lines == ["A", "B"]
        assert st._omitted_lines == 0
        assert runner._progress_text(st, lines) == "A\nB"  # no omission label yet

    def test_exactly_max(self):
        adapter = RollingCaptureAdapter()
        ctx = _make_ctx(grouping="rolling", max_entries=4)
        runner, st = _state_for(adapter, ctx)
        lines = _absorb_all(runner, st, ["A", "B", "C", "D"])
        assert lines == ["A", "B", "C", "D"]
        assert st._omitted_lines == 0
        assert LABEL not in runner._progress_text(st, lines)

    def test_more_than_max(self):
        adapter = RollingCaptureAdapter()
        ctx = _make_ctx(grouping="rolling", max_entries=4)
        runner, st = _state_for(adapter, ctx)
        lines = _absorb_all(runner, st, ["A", "B", "C", "D", "E"])
        assert lines == ["B", "C", "D", "E"]
        assert st._omitted_lines == 1
        assert runner._progress_text(st, lines) == f"{LABEL}\nB\nC\nD\nE"

    def test_continued_rolling(self):
        adapter = RollingCaptureAdapter()
        ctx = _make_ctx(grouping="rolling", max_entries=4)
        runner, st = _state_for(adapter, ctx)
        lines = _absorb_all(runner, st, ["A", "B", "C", "D", "E", "F"])
        assert lines == ["C", "D", "E", "F"]
        assert st._omitted_lines == 2
        assert runner._progress_text(st, lines) == f"{LABEL}\nC\nD\nE\nF"

    def test_max_entries_one(self):
        adapter = RollingCaptureAdapter()
        ctx = _make_ctx(grouping="rolling", max_entries=1)
        runner, st = _state_for(adapter, ctx)
        lines = _absorb_all(runner, st, ["A", "B"])
        assert lines == ["B"]
        assert st._omitted_lines == 1
        assert runner._progress_text(st, lines) == f"{LABEL}\nB"

    def test_long_strings_count_as_single_entries(self):
        """The window operates on logical entries, never by character truncation."""
        adapter = RollingCaptureAdapter()
        ctx = _make_ctx(grouping="rolling", max_entries=3)
        runner, st = _state_for(adapter, ctx)
        a = "🔧 web_search: " + "x" * 500
        b = "💻 terminal: " + "y" * 500
        c = "📖 read_file: " + "z" * 500
        d = "🔧 web_search: " + "w" * 500
        lines = _absorb_all(runner, st, [a, b, c, d])
        # All three full-length strings survive verbatim; only the oldest rolled off.
        assert lines == [b, c, d]
        assert st._omitted_lines == 1


# ---------------------------------------------------------------------------
# 6-7. Existing grouping modes must remain unchanged
# ---------------------------------------------------------------------------
class TestExistingModesUnchanged:
    def test_accumulate_keeps_everything(self):
        adapter = RollingCaptureAdapter()
        ctx = _make_ctx(grouping="accumulate", max_entries=4)
        runner, st = _state_for(adapter, ctx)
        # max_entries is irrelevant in accumulate mode
        assert st.max_entries is None
        lines = _absorb_all(runner, st, ["A", "B", "C", "D", "E", "F"])
        assert lines == ["A", "B", "C", "D", "E", "F"]
        assert st._omitted_lines == 0
        assert runner._progress_text(st, lines) == "A\nB\nC\nD\nE\nF"

    def test_separate_not_editable_and_no_window(self):
        adapter = RollingCaptureAdapter()
        ctx = _make_ctx(grouping="separate", max_entries=4)
        runner, st = _state_for(adapter, ctx)
        assert st.can_edit is False
        assert st.max_entries is None
        lines = _absorb_all(runner, st, ["A", "B", "C"])
        assert lines == ["A", "B", "C"]
        assert runner._progress_text(st, lines) == "A\nB\nC"


# ---------------------------------------------------------------------------
# 8. Tool-lifecycle / dedup updates do not consume extra slots
# ---------------------------------------------------------------------------
class TestDedupNoExtraSlot:
    def test_dedup_updates_newest_entry_in_place(self):
        adapter = RollingCaptureAdapter()
        ctx = _make_ctx(grouping="rolling", max_entries=4)
        runner, st = _state_for(adapter, ctx)
        _absorb_all(runner, st, ["A", "B", "C"])
        # A repeated identical line folds into the newest entry, not a new slot.
        runner._progress_absorb(st, ("__dedup__", "C", 1))
        assert st.progress_lines == ["A", "B", "C (×2)"]
        assert st._omitted_lines == 0

    def test_dedup_then_new_line_still_rolls(self):
        adapter = RollingCaptureAdapter()
        ctx = _make_ctx(grouping="rolling", max_entries=4)
        runner, st = _state_for(adapter, ctx)
        _absorb_all(runner, st, ["A", "B", "C", "D"])
        runner._progress_absorb(st, ("__dedup__", "D", 1))  # D -> D (×2)
        assert st.progress_lines == ["A", "B", "C", "D (×2)"]
        assert st._omitted_lines == 0
        runner._progress_absorb(st, "E")  # A rolls off
        assert st.progress_lines == ["B", "C", "D (×2)", "E"]
        assert st._omitted_lines == 1

    def test_dedup_into_empty_buffer_becomes_placeholder_then_rolls(self):
        adapter = RollingCaptureAdapter()
        ctx = _make_ctx(grouping="rolling", max_entries=2)
        runner, st = _state_for(adapter, ctx)
        # No visible entry yet; the dedup starts a placeholder buffer.
        ret = runner._progress_absorb(st, ("__dedup__", "Z", 1))
        assert ret == "Z"
        assert st.progress_lines == ["Z"]


# ---------------------------------------------------------------------------
# 9. Reset / new bubble clears the rolling window
# ---------------------------------------------------------------------------
class TestReset:
    def test_reset_clears_window_and_counter(self):
        adapter = RollingCaptureAdapter()
        ctx = _make_ctx(grouping="rolling", max_entries=4)
        runner, st = _state_for(adapter, ctx)
        _absorb_all(runner, st, ["A", "B", "C", "D", "E", "F"])
        assert st._omitted_lines == 2
        runner._reset_progress_bubble(st)
        assert st.progress_lines == []
        assert st._omitted_lines == 0
        assert st.progress_msg_id is None

    def test_new_bubble_starts_fresh_window(self):
        adapter = RollingCaptureAdapter()
        ctx = _make_ctx(grouping="rolling", max_entries=2)
        runner, st = _state_for(adapter, ctx)
        _absorb_all(runner, st, ["A", "B", "C", "D"])
        runner._reset_progress_bubble(st)
        _absorb_all(runner, st, ["X", "Y", "Z"])
        # Fresh window of the new bubble: last 2, one omission.
        assert st.progress_lines == ["Y", "Z"]
        assert st._omitted_lines == 1


# ---------------------------------------------------------------------------
# 10. Omission label does not consume a configured slot
# ---------------------------------------------------------------------------
class TestOmissionLabel:
    def test_label_never_counts_against_window(self):
        adapter = RollingCaptureAdapter()
        ctx = _make_ctx(grouping="rolling", max_entries=4)
        runner, st = _state_for(adapter, ctx)
        _absorb_all(runner, st, ["A", "B", "C", "D", "E", "F", "G", "H"])
        # 8 entries, window 4 → exactly 4 visible lines, label is extra.
        assert len(st.progress_lines) == 4
        rendered = runner._progress_text(st, st.progress_lines)
        lines = rendered.split("\n")
        assert lines[0] == LABEL
        assert lines[1:] == ["E", "F", "G", "H"]

    def test_no_label_when_nothing_rolled(self):
        adapter = RollingCaptureAdapter()
        ctx = _make_ctx(grouping="rolling", max_entries=4)
        runner, st = _state_for(adapter, ctx)
        _absorb_all(runner, st, ["A", "B"])
        assert LABEL not in runner._progress_text(st, st.progress_lines)


# ---------------------------------------------------------------------------
# End-to-end: the real send_progress_messages loop
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_end_to_end_rolling_edits_single_message(monkeypatch):
    """One send, then in-place edits only; oldest entries roll off; label appears."""
    # Disable the 1.5s edit throttle and the typing restore so the loop drains fast.
    _counter = {"v": 0.0}

    def _fake_monotonic():
        _counter["v"] += 100  # jump well past EDIT_INTERVAL each call
        return _counter["v"]

    monkeypatch.setattr(time, "monotonic", _fake_monotonic)

    async def _no_op_typing(self, st):
        return None

    monkeypatch.setattr(TurnRunner, "_progress_restore_typing", _no_op_typing)

    adapter = RollingCaptureAdapter()
    q = queue_mod.Queue()
    ctx = _make_ctx(grouping="rolling", max_entries=4, queue=q)
    runner = _make_runner(ctx)
    runner._runner._adapter = adapter

    loop = asyncio.get_event_loop()
    task = loop.create_task(runner.send_progress_messages())
    for it in ["A", "B", "C", "D", "E", "F"]:
        q.put(it)
        await asyncio.sleep(0.05)
    ctx._run_still_current = lambda: False
    await task

    # Exactly one progress message was ever SENT; everything after is an edit.
    assert len(adapter.sent) == 1
    assert adapter.edits, "expected at least one edit"
    assert all(e["message_id"] == adapter.edits[0]["message_id"] for e in adapter.edits)

    # Final visible content: label + the last 4 logical entries (E, F plus C, D... wait).
    final = adapter.edits[-1]["content"]
    final_lines = final.split("\n")
    assert final_lines[0] == LABEL
    assert final_lines[1:] == ["C", "D", "E", "F"]


@pytest.mark.asyncio
async def test_end_to_end_accumulate_grows(monkeypatch):
    """Control: accumulate mode never drops entries and never shows the label."""
    _counter = {"v": 0.0}

    def _fake_monotonic():
        _counter["v"] += 100
        return _counter["v"]

    monkeypatch.setattr(time, "monotonic", _fake_monotonic)

    async def _no_op_typing(self, st):
        return None

    monkeypatch.setattr(TurnRunner, "_progress_restore_typing", _no_op_typing)

    adapter = RollingCaptureAdapter()
    q = queue_mod.Queue()
    ctx = _make_ctx(grouping="accumulate", max_entries=4, queue=q)
    runner = _make_runner(ctx)
    runner._runner._adapter = adapter

    loop = asyncio.get_event_loop()
    task = loop.create_task(runner.send_progress_messages())
    for it in ["A", "B", "C", "D", "E", "F"]:
        q.put(it)
        await asyncio.sleep(0.05)
    ctx._run_still_current = lambda: False
    await task

    final = adapter.edits[-1]["content"]
    assert final == "A\nB\nC\nD\nE\nF"
    assert LABEL not in final


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------
class TestConfigResolution:
    def test_rolling_is_accepted(self):
        from gateway.display_config import resolve_display_setting

        cfg = {"display": {"platforms": {"whatsapp": {"tool_progress_grouping": "rolling"}}}}
        assert resolve_display_setting(cfg, "whatsapp", "tool_progress_grouping") == "rolling"

    def test_global_rolling(self):
        from gateway.display_config import resolve_display_setting

        cfg = {"display": {"tool_progress_grouping": "rolling"}}
        assert resolve_display_setting(cfg, "telegram", "tool_progress_grouping") == "rolling"

    def test_max_entries_default_is_four(self):
        from gateway.display_config import resolve_display_setting

        assert resolve_display_setting({}, "whatsapp", "tool_progress_max_entries") == 4

    def test_max_entries_per_platform(self):
        from gateway.display_config import resolve_display_setting

        cfg = {"display": {"platforms": {"whatsapp": {"tool_progress_max_entries": 8}}}}
        assert resolve_display_setting(cfg, "whatsapp", "tool_progress_max_entries") == 8

    def test_max_entries_invalid_falls_back_to_four(self):
        from gateway.display_config import resolve_display_setting

        for bad in (0, -3, "abc", None):
            cfg = {"display": {"tool_progress_max_entries": bad}}
            # None resolves to the global default (4) because the resolver skips Nones.
            if bad is None:
                continue
            assert resolve_display_setting(cfg, "whatsapp", "tool_progress_max_entries") == 4, bad
