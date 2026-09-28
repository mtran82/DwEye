"""TUI: prompt_toolkit layer over controller.Controller.

Keys:
  printable  append to fragment, refresh suggestions
  space      accept selected suggestion + space (compound fixes "a"->"an")
  right      bypass: insert fragment verbatim + space
  up/down    move selection (new keystroke resets to #1)
  enter      accept selection, no trailing space, quit
  backspace  delete last fragment char (committed words immutable)
  esc / c-c  quit
"""
import os
import sys

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import Controller  # noqa: E402
from pipeline import Pipeline  # noqa: E402

MAX_SUGGESTIONS = 5


class TuiApp:
    def __init__(self, ctrl: Controller):
        self.ctrl = ctrl
        self.text_ctrl = FormattedTextControl(self._render_text)
        self.list_ctrl = FormattedTextControl(self._render_list)
        self.status_ctrl = FormattedTextControl(self._render_status)

        layout = Layout(HSplit([
            Window(self.text_ctrl, wrap_lines=True),
            Window(height=1),
            Window(self.list_ctrl, height=MAX_SUGGESTIONS),
            Window(height=1),
            Window(self.status_ctrl, height=1)]))

        kb = KeyBindings()

        @kb.add("escape")
        @kb.add("c-c")
        def _quit(event):
            event.app.exit()

        for ch in "abcdefghijklmnopqrstuvwxyz'":
            @kb.add(ch)
            def _char(event, ch=ch):
                self.ctrl.on_char(ch)

        @kb.add("space")
        def _space(event):
            self.ctrl.accept()

        @kb.add(".")
        def _period(event):
            self.ctrl.on_period()

        @kb.add("right")
        def _bypass(event):
            self.ctrl.bypass()

        @kb.add("enter")
        def _enter(event):
            result = self.ctrl.accept(finish=True)
            event.app.exit(result=result)

        @kb.add("backspace")
        def _bs(event):
            self.ctrl.on_backspace()

        for digit in "12345":
            @kb.add(digit)
            def _digit(event, d=digit):
                i = int(d) - 1
                if i < len(self.ctrl.suggestions):
                    self.ctrl.selected = i
                    self.ctrl.accept()

        @kb.add("up")
        def _up(event):
            self.ctrl.move_selection(-1)

        @kb.add("down")
        def _down(event):
            self.ctrl.move_selection(1)

        self.app = Application(layout=layout, key_bindings=kb,
                               full_screen=True)

    # ---- renderers ----
    def _render_text(self):
        return [("bold", self.ctrl.get_text()), ("", "\u2588")]

    def _render_list(self):
        if not self.ctrl.suggestions:
            return [("", "(no suggestions - \u2192 to bypass)")]
        rows = []
        for i, sug in enumerate(self.ctrl.suggestions):
            display = sug["compound"] or sug["word"]
            prob = sug["prob"]
            mark = "> " if i == self.ctrl.selected else "  "
            style = "bold fg:ansigreen" if i == self.ctrl.selected else ""
            suffix = " (fixes a->an)" if sug["compound"] else ""
            rows.append((style, f"{mark}{display:<16} {prob*100:4.0f}%{suffix}\n"))
        return rows

    def _render_status(self):
        a, b = self.ctrl.timings
        return [("", f"symspell {a:.1f}ms | ngram {b:.1f}ms | {self.ctrl.status}")]

    def run(self):
        self.ctrl.refresh()
        return self.app.run()


def main():
    ctrl = Controller(Pipeline())
    result = TuiApp(ctrl).run()
    if result:
        print("\nfinal text:", result)


if __name__ == "__main__":
    main()