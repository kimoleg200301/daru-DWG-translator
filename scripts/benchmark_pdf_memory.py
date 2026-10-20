"""Measure peak process memory for synthetic scanned PDFs, without network calls.

Run each page count in a fresh process. --package-path can point to an older
checkout's src directory for a like-for-like comparison.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def peak_memory_mib() -> float:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        read = ctypes.windll.psapi.GetProcessMemoryInfo
        read.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not read(process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return counters.PeakWorkingSetSize / 1024**2
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**2 if sys.platform == "darwin" else 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=12)
    parser.add_argument(
        "--package-path", type=Path, default=Path(__file__).resolve().parents[1] / "src"
    )
    args = parser.parse_args()
    if args.pages < 1:
        parser.error("--pages must be positive")
    sys.path.insert(0, str(args.package_path))
    from PIL import Image, ImageDraw
    from daru.pdf import _core

    def pages(_loader, _path):
        for index in range(args.pages):
            image = Image.new("RGB", (1240, 1754), "white")
            ImageDraw.Draw(image).text(
                (50, 65), "Inspect the safety valve before operation.", fill="black"
            )
            yield _core.PageImage(index, image, 1240, 1754, 150, (595.2, 841.92))

    class Extractor:
        def __init__(self, *args, **kwargs):
            pass

        def analyze(self, page):
            block = _core.BlockRegion(
                page.page_index,
                str(page.page_index),
                "TEXT",
                (45, 55, 800, 105),
                source_text="Inspect the safety valve before operation.",
            )
            return _core.TextractPageData(page, {}, {}, [block])

    class Engine:
        def __init__(self, **kwargs):
            pass

        def backend_name(self):
            return "synthetic"

        def set_drawing_context(self, texts):
            pass

        def translate_many(self, texts):
            return ["Check the safety valve." for _ in texts]

    _core.PageLoader.iter_pages = pages
    _core.PageLoader.load = lambda self, path: list(pages(self, path))
    _core.TextractLayoutExtractor = Extractor
    _core.TranslationEngine = Engine
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="daru-memory-") as directory:
        root = Path(directory)
        source = root / "source.pdf"
        source.write_bytes(b"synthetic fixture")
        _core.translate_pdf(
            input_path=source,
            output_path=root / "output.pdf",
            translator_name="noop",
            source_lang="en",
            target_lang="ru",
            log=lambda _message: None,
            checkpoint_path=root / "checkpoint.json",
            layer_json_path=root / "layer.json",
            processing_options={"render_dpi": 150, "dpi": 150},
        )
    print(
        json.dumps(
            {
                "pages": args.pages,
                "peak_mib": round(peak_memory_mib(), 1),
                "seconds": round(time.perf_counter() - started, 2),
            }
        )
    )


if __name__ == "__main__":
    main()
