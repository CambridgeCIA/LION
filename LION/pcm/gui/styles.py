"""Reusable Qt styles for the PCM GUI."""

BLUE_SCROLLBAR_STYLE = """
QScrollBar:vertical {
    background: #e8eef7;
    width: 14px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #3b82f6;
    min-height: 30px;
    border-radius: 7px;
}

QScrollBar::handle:vertical:hover {
    background: #2563eb;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #e8eef7;
    height: 14px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background: #3b82f6;
    min-width: 30px;
    border-radius: 7px;
}

QScrollBar::handle:horizontal:hover {
    background: #2563eb;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0px;
}
"""
