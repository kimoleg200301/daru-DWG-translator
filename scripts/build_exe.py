"""Interactive GUI build with application version selection."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = PROJECT_ROOT / "src" / "daru" / "__init__.py"
SPEC_FILE = PROJECT_ROOT / "daru_gui.spec"
VERSION_PATTERN = re.compile(
    r'(?m)^__version__\s*=\s*["\'](?P<version>[^"\']+)["\']\s*$'
)
VALID_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:\.\d+)?$")


def read_version(path: Path = VERSION_FILE) -> str:
    content = path.read_bytes().decode("utf-8")
    match = VERSION_PATTERN.search(content)
    if match is None:
        raise RuntimeError(f"Не найдена переменная __version__ в {path}")
    return match.group("version")


def validate_version(value: str) -> str:
    version = value.strip()
    if not VALID_VERSION_PATTERN.fullmatch(version):
        raise ValueError("Версия должна иметь формат X.Y.Z или X.Y.Z.W")
    if any(int(part) > 65535 for part in version.split(".")):
        raise ValueError("Каждая часть версии должна быть не больше 65535")
    return version


def write_version(version: str, path: Path = VERSION_FILE) -> None:
    validated = validate_version(version)
    content = path.read_bytes().decode("utf-8")
    updated, replacements = VERSION_PATTERN.subn(
        f'__version__ = "{validated}"',
        content,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError(f"Не удалось обновить __version__ в {path}")
    if updated != content:
        path.write_bytes(updated.encode("utf-8"))


def prompt_version(current_version: str) -> str:
    while True:
        entered = input(
            f"Введите версию сборки [{current_version}]: "
        ).strip()
        try:
            return validate_version(entered or current_version)
        except ValueError as exc:
            print(f"Ошибка: {exc}")


def build(version: str) -> int:
    write_version(version)
    print(f"Версия приложения: {version}")
    print("Запуск PyInstaller...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            str(SPEC_FILE),
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode == 0:
        artifact = (
            PROJECT_ROOT
            / "dist"
            / "DaruDocumentTranslator"
            / "DaruDocumentTranslator.exe"
        )
        print(f"Сборка завершена: {artifact}")
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сборка Daru Document Translator с выбором версии."
    )
    parser.add_argument(
        "--version",
        help="Версия без интерактивного запроса, например 1.1.0.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current_version = read_version()
    requested_version = args.version.strip() if args.version is not None else None
    if requested_version:
        version = validate_version(requested_version)
    elif args.version is not None:
        version = current_version
    else:
        version = prompt_version(current_version)
    return build(version)


if __name__ == "__main__":
    raise SystemExit(main())
