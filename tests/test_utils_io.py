"""Tests for daru.utils.io — file I/O helpers."""

from pathlib import Path

from daru.utils.io import ensure_parent, paths_equal, write_map_csv, read_map_csv, write_translated_txt


class TestEnsureParent:
    def test_creates_parent(self, tmp_path):
        target = tmp_path / "a" / "b" / "file.txt"
        ensure_parent(target)
        assert target.parent.exists()

    def test_existing_parent_ok(self, tmp_path):
        target = tmp_path / "file.txt"
        ensure_parent(target)  # should not raise


class TestPathsEqual:
    def test_same_path(self, tmp_path):
        p = tmp_path / "file.txt"
        assert paths_equal(p, p)

    def test_different_paths(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        assert not paths_equal(a, b)


class TestMapCsv:
    def test_write_and_read(self, tmp_path):
        csv_path = tmp_path / "map.csv"
        english = ["hello", "world"]
        russian = ["привет", "мир"]
        write_map_csv(csv_path, english, russian)

        result = read_map_csv(csv_path)
        assert result == {"hello": "привет", "world": "мир"}

    def test_read_missing_file(self, tmp_path):
        result = read_map_csv(tmp_path / "no_such.csv")
        assert result == {}

    def test_read_none_path(self):
        result = read_map_csv(None)
        assert result == {}

    def test_empty_csv(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("text_en,text_ru\n", encoding="utf-8")
        result = read_map_csv(csv_path)
        assert result == {}


class TestWriteTranslatedTxt:
    def test_basic(self, tmp_path):
        out = tmp_path / "translated.txt"
        items = [("hello", 3), ("world", 1)]
        translations = ["привет", "мир"]
        write_translated_txt(out, items, translations)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines == ["[3] привет", "[1] мир"]
