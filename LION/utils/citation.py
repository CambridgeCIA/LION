# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""Citations of where LION's components come from (a model's paper, a
solver's, a dataset's): each component's ``cite(cite_format="bib")`` returns
its citation as text, BibTeX or MLA, or None if it has none."""

import textwrap

FORMATS = ("bib", "MLA")


def citation(cite_format, bib, mla):
    """``bib`` or ``mla``, as ``cite_format`` asks, without indentation."""
    if cite_format not in FORMATS:
        raise ValueError(
            f'cite_format "{cite_format}" is not understood: one of {", ".join(FORMATS)}'
        )
    text = bib if cite_format == "bib" else mla
    return textwrap.dedent(text).strip()
