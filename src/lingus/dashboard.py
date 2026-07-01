"""Rich terminal dashboard — the "you can't tune personality blind" instrument.

Renders the arbiter live: salience score against the dynamic threshold (the
whole point — you watch the bar move), which signals fired, mood drift, and a
running log of speak/drop decisions, alongside the stream context that drove
them. Implements the `Monitor` protocol; enabled with `lingus --dashboard`.

`rich` is imported lazily and only here, so it stays an optional extra:
    pip install -e ".[dashboard]"
"""

from __future__ import annotations

from collections import deque

from .monitor import TickReport, format_clock, reason_kind

_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], lo: float, hi: float) -> str:
    if not values:
        return ""
    span = (hi - lo) or 1.0
    out = []
    for value in values:
        idx = int((value - lo) / span * (len(_SPARK_BLOCKS) - 1))
        out.append(_SPARK_BLOCKS[max(0, min(len(_SPARK_BLOCKS) - 1, idx))])
    return "".join(out)


def _mood_bar(value: float, width: int = 10) -> str:
    """Bipolar bar in [-1, 1]; centre is baseline."""
    filled = int(round((value + 1.0) / 2.0 * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "·" * (width - filled)


class RichDashboard:
    """Live full-screen terminal view of the cognition loop. Monitor protocol."""

    def __init__(self, persona_name: str, platform: str, history: int = 48) -> None:
        # Lazy imports: this module is only loaded when --dashboard is passed.
        from rich.console import Console
        from rich.live import Live

        self._console = Console()
        self._live = Live(
            console=self._console,
            screen=True,
            auto_refresh=False,
            transient=True,
        )
        self.persona_name = persona_name
        self.platform = platform
        self._scores: deque[float] = deque(maxlen=history)
        self._thresholds: deque[float] = deque(maxlen=history)
        self._decisions: deque[str] = deque(maxlen=10)
        self._latest: TickReport | None = None
        self._ticks = 0
        self._start_t: float | None = None

    # --- Monitor protocol ---
    def start(self) -> None:
        self._live.start(refresh=True)

    def stop(self) -> None:
        self._live.stop()

    def on_tick(self, report: TickReport) -> None:
        if self._start_t is None:
            self._start_t = report.t
        self._ticks += 1
        self._latest = report
        self._scores.append(report.decision.score)
        self._thresholds.append(report.decision.threshold)
        clock = format_clock(report.t - self._start_t)
        if report.posted:
            self._decisions.appendleft(f"[green]{clock} ✓ posted[/]  {report.posted}")
        elif report.dropped:
            self._decisions.appendleft(f"[yellow]{clock} ✗ dropped[/] {report.dropped}")
        self._live.update(self._render(), refresh=True)

    # --- rendering ---
    def _render(self):
        from rich.layout import Layout

        layout = Layout()
        layout.split_column(
            Layout(self._header(), name="header", size=1),
            Layout(name="body"),
            Layout(self._decision_log(), name="log", size=12),
        )
        layout["body"].split_row(
            Layout(self._arbiter_panel(), name="arbiter"),
            Layout(self._stream_panel(), name="stream"),
        )
        return layout

    def _header(self):
        from rich.text import Text

        uptime = (
            "00:00"
            if self._start_t is None or self._latest is None
            else (format_clock(self._latest.t - self._start_t))
        )
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append("  Lingus ", style="bold magenta")
        text.append(f"· {self.persona_name} ", style="bold")
        text.append(f"· {self.platform}", style="dim")
        text.append(f"   uptime {uptime} · tick {self._ticks}", style="dim")
        return text

    def _arbiter_panel(self):
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        report = self._latest
        lo = min([*self._scores, *self._thresholds, 0.0])
        hi = max([*self._scores, *self._thresholds, 1.0])

        grid = Table.grid(padding=(0, 1))
        grid.add_column(justify="right", style="bold")
        grid.add_column()

        if report is None:
            grid.add_row("status", Text("waiting for first tick…", style="dim"))
            return Panel(grid, title="arbiter", border_style="cyan")

        decision = report.decision
        speak = decision.should_reply
        verdict = Text("SPEAK ✓" if speak else "hold", style="bold green" if speak else "dim")

        grid.add_row("verdict", verdict)
        grid.add_row(
            "score",
            Text(f"{decision.score:5.2f}  ", style="bold")
            + Text(_sparkline(list(self._scores), lo, hi), style="green"),
        )
        grid.add_row(
            "thresh",
            Text(f"{decision.threshold:5.2f}  ", style="bold")
            + Text(_sparkline(list(self._thresholds), lo, hi), style="red"),
        )
        grid.add_row("reasons", self._reasons_text(decision.reasons))
        grid.add_row(
            "mood",
            Text(f"{report.mood:+.2f} ", style="bold")
            + Text(f"[{_mood_bar(report.mood)}]", style="cyan"),
        )
        grid.add_row("events", Text(str(report.n_events), style="dim"))
        return Panel(grid, title="arbiter", border_style="cyan")

    def _reasons_text(self, reasons: list[str]):
        from rich.text import Text

        if not reasons:
            return Text("—", style="dim")
        text = Text()
        for i, reason in enumerate(reasons):
            if i:
                text.append(" ")
            kind = reason_kind(reason)
            if kind == "positive":
                text.append(reason, style="bold green")
            elif kind == "blocking":
                text.append(reason, style="bold red")
            else:  # cooldown and other informational reasons
                text.append(reason, style="yellow")
        return text

    def _stream_panel(self):
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        report = self._latest
        grid = Table.grid(padding=(0, 1))
        grid.add_column(justify="right", style="bold")
        grid.add_column(ratio=1, overflow="fold")
        if report is None:
            grid.add_row("", Text("…", style="dim"))
            return Panel(grid, title="stream", border_style="blue")

        grid.add_row("scene", Text(report.scene_summary or "—", style="dim"))
        grid.add_row("speech", Text(report.transcript_tail or "—"))
        chat = Text()
        for line in report.recent_chat[-6:]:
            chat.append(f"{line.author}: ", style="bold blue")
            chat.append(f"{line.text}\n")
        grid.add_row("chat", chat if report.recent_chat else Text("—", style="dim"))
        return Panel(grid, title="stream", border_style="blue")

    def _decision_log(self):
        from rich.panel import Panel
        from rich.text import Text

        if not self._decisions:
            body = Text("no posts yet — watching", style="dim")
        else:
            body = Text("\n").join(Text.from_markup(line) for line in self._decisions)
        return Panel(body, title="decisions", border_style="green")
