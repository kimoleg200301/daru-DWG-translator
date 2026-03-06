"""ODA File Converter discovery and conversion helpers."""

import os
from pathlib import Path
from typing import Callable

from ezdxf import options as ezdxf_options
from ezdxf.addons import odafc
from ezdxf.addons.odafc import ODAFCNotInstalledError, UnknownODAFCError


def ensure_odafc_config() -> None:
    if odafc.is_installed():
        return

    candidates = []
    env_exec = os.environ.get("ODA_FILE_CONVERTER")
    if env_exec:
        candidates.append(Path(env_exec))
    env_path = os.environ.get("ODAFC_PATH") or os.environ.get("ODAFC_HOME")
    if env_path:
        candidates.append(Path(env_path) / "ODAFileConverter.exe")

    if os.name == "nt":
        program_files = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
        for base in program_files:
            if not base:
                continue
            oda_dir = Path(base) / "ODA"
            if oda_dir.exists():
                for sub in sorted(oda_dir.glob("ODAFileConverter*"), reverse=True):
                    candidates.append(sub / "ODAFileConverter.exe")

    for candidate in candidates:
        if candidate and candidate.exists():
            ezdxf_options.set("odafc-addon", "win_exec_path", str(candidate))
            if odafc.is_installed():
                return


def convert_with_odafc(source: Path, dest: Path, logger: Callable[[str], None], *, replace: bool = True) -> None:
    ensure_odafc_config()
    try:
        odafc.convert(str(source), str(dest), version="R2018", replace=replace)
    except ODAFCNotInstalledError as exc:
        raise RuntimeError(
            "Не найден ODA File Converter. Установите ODAFileConverter и добавьте его в PATH."
        ) from exc
    except UnknownODAFCError as exc:
        raise RuntimeError(f"Ошибка ODA File Converter: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Не удалось конвертировать файл через ODA FC: {exc}") from exc
    else:
        logger(f"Конвертация выполнена: {dest}")
