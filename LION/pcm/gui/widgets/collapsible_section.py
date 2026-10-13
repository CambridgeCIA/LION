"""Simple collapsible section widget."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """A section with a clickable title that shows or hides its contents."""

    def __init__(
        self, title: str, *, expanded: bool = False, parent: QWidget | None = None
    ) -> None:
        """Initialize the collapsible section."""
        super().__init__(parent)

        self.toggle_button = QToolButton(text=title, checkable=True, checked=expanded)
        self.toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.toggle_button.toggled.connect(self._set_expanded)

        self.content = QWidget()
        self.content.setVisible(expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content)

    def set_content_layout(self, content_layout) -> None:
        """Set the layout displayed inside the section."""
        self.content.setLayout(content_layout)

    def _set_expanded(self, expanded: bool) -> None:
        """Show or hide the section contents."""
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.content.setVisible(expanded)
