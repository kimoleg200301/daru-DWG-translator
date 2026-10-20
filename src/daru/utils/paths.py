"""User-writable runtime paths; importing this module creates no files."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable


def user_data_dir() -> Path:
    configured = os.environ.get("DARU_DATA_DIR", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".daru"


def migrate_pdf_cache(
    legacy: Path, destination: Path, log: Callable[[str], None]
) -> None:
    """Copy known legacy cache files on demand, preserving both existing copies."""
    if not legacy.is_dir() or legacy.resolve() == destination.resolve():
        return
    candidates = [legacy / "textract_requests.log", legacy / "vision_requests.json"]
    candidates.extend(legacy.glob("*.translation.json"))
    for source in candidates:
        target = destination / source.name
        if not source.is_file() or target.exists():
            continue
        temporary = None
        try:
            destination.mkdir(parents=True, exist_ok=True)
            with (
                source.open("rb") as incoming,
                tempfile.NamedTemporaryFile(dir=destination, delete=False) as outgoing,
            ):
                temporary = Path(outgoing.name)
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            # Publish the complete file only if another job has not created it.
            os.link(temporary, target)
        except FileExistsError:
            continue
        except OSError as exc:
            log(f"Не удалось перенести кэш {source.name}: {exc}")
        else:
            log(f"Кэш скопирован в пользовательский каталог: {target}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
