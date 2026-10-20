"""Disk-backed page images with explicit, bounded image lifetimes."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

from .models import PageImage

if TYPE_CHECKING:
    from PIL import Image


class StoredPage(PageImage):
    def __init__(self, page: PageImage, path: Path) -> None:
        for item in fields(PageImage):
            if item.name != "image":
                setattr(self, item.name, getattr(page, item.name))
        self.path = path
        self._loaded: Image.Image | None = None

    @property
    def image(self) -> Image.Image:
        if self._loaded is None:
            from PIL import Image

            with Image.open(self.path) as image:
                self._loaded = image.copy()
        return self._loaded

    def release(self) -> None:
        if self._loaded is not None:
            self._loaded.close()
            self._loaded = None


def release_page(page: PageImage) -> None:
    if isinstance(page, StoredPage):
        page.release()


class PageStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.pages: list[StoredPage] = []

    def add(self, page: PageImage) -> StoredPage:
        path = self.root / f"{len(self.pages):06d}.png"
        page.image.save(path, format="PNG", compress_level=1)
        stored = StoredPage(page, path)
        self.pages.append(stored)
        return stored

    def close(self) -> None:
        for page in self.pages:
            page.release()


class ImageSpool(Sequence["Image.Image"]):
    """Export images whose decoded pixels live only during one iteration."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.paths: list[Path] = []

    def append(self, image: Image.Image) -> None:
        path = self.root / f"{len(self.paths):06d}.png"
        image.save(path, format="PNG", compress_level=1)
        self.paths.append(path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image

        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        with Image.open(self.paths[index]) as image:
            return image.copy()

    def __iter__(self) -> Iterator[Image.Image]:
        for index in range(len(self)):
            image = self[index]
            try:
                yield image
            finally:
                image.close()
