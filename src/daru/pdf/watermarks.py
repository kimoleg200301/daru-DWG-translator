"""Keep explicitly marked PDF watermark Forms outside text processing."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from uuid import uuid4


def _marked_form_xrefs(page: Any) -> list[int]:
    document = page.parent
    return list(dict.fromkeys(
        xref for xref, *_ in page.get_xobjects()
        if document.xref_get_key(xref, "PieceInfo/ADBE_CompoundType/Private")
        == ("name", "/Watermark")
    ))


@contextmanager
def preserve_watermarks(page: Any) -> Iterator[None]:
    """Hide Adobe Watermark Forms during extraction or text redaction.

    MuPDF clones Forms while applying redactions and discards resources of
    empty Forms. Restore both the full object and its stream, including any
    clones, so original fonts, opacity, transforms and shared Forms survive.
    Ordinary text and Forms without an explicit watermark marker stay live.
    """
    document = page.parent
    snapshots = {
        xref: (document.xref_object(xref), document.xref_stream(xref))
        for xref in _marked_form_xrefs(page)
    }
    if not snapshots:
        yield
        return

    marker = f"DaruWatermarkPreserve{uuid4().hex}"
    try:
        for xref in snapshots:
            document.xref_set_key(xref, marker, str(xref))
            document.update_stream(xref, b"")
        yield
    finally:
        # A Form can be shared by several pages. Restore originals even when
        # the current page has switched its Do operator to a cloned Form.
        targets = {xref: xref for xref in snapshots}
        try:
            for xref, *_ in page.get_xobjects():
                kind, value = document.xref_get_key(xref, marker)
                if kind == "int" and int(value) in snapshots:
                    targets[xref] = int(value)
        finally:
            for target, original in targets.items():
                obj, stream = snapshots[original]
                document.update_object(target, obj)
                document.update_stream(target, stream)


__all__ = ["preserve_watermarks"]
