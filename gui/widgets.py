"""Reusable Tk widgets for the ROSA GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Iterable, Optional


class LabeledSeparator(ttk.Frame):
    def __init__(self, master: tk.Widget, text: str, padding: int = 4, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.columnconfigure(1, weight=1)
        ttk.Label(self, text=text).grid(row=0, column=0, padx=(0, padding))
        ttk.Separator(self, orient="horizontal").grid(row=0, column=1, sticky="ew")


class LabeledEntry(ttk.Frame):
    def __init__(
        self,
        master: tk.Widget,
        label: str,
        textvariable: Optional[tk.Variable] = None,
        width: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        ttk.Label(self, text=label).pack(side="left", padx=(0, 4))
        self.entry = ttk.Entry(self, textvariable=textvariable, width=width)
        self.entry.pack(side="left", fill="x", expand=True)

    def get(self) -> str:
        return self.entry.get()

    def configure_entry(self, **kwargs) -> None:
        self.entry.configure(**kwargs)


class LabeledOptionMenu(ttk.Frame):
    def __init__(
        self,
        master: tk.Widget,
        label: str,
        variable: tk.Variable,
        options: Iterable[str],
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        ttk.Label(self, text=label).pack(side="left", padx=(0, 4))
        self.option = ttk.OptionMenu(self, variable, variable.get(), *options)
        self.option.pack(side="left")


class ToggleButton(ttk.Button):
    def __init__(self, master: tk.Widget, text_on: str, text_off: str, command: Callable[[bool], None], **kwargs):
        self.state = tk.BooleanVar(value=False)
        self.text_on = text_on
        self.text_off = text_off
        self.user_command = command
        super().__init__(master, text=self.text_off, command=self._toggle, **kwargs)

    def _toggle(self) -> None:
        new_state = not self.state.get()
        self.state.set(new_state)
        self.configure(text=self.text_on if new_state else self.text_off)
        self.user_command(new_state)

    def reset(self) -> None:
        self.state.set(False)
        self.configure(text=self.text_off)


__all__ = [
    "LabeledEntry",
    "LabeledOptionMenu",
    "LabeledSeparator",
    "ToggleButton",
]
