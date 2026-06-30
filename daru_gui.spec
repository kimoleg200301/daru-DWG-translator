# -*- mode: python ; coding: utf-8 -*-

import re
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

project_root = Path(SPECPATH)
sys.path.insert(0, str(project_root / 'src'))

from daru import APP_EXECUTABLE_NAME, APP_NAME, __version__

block_cipher = None
daru_hiddenimports = collect_submodules('daru')


def normalized_version(value):
    parts = [int(part) for part in re.findall(r'\d+', value)[:4]]
    return tuple((parts + [0, 0, 0, 0])[:4])


version_info = None
if sys.platform == 'win32':
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo,
        StringFileInfo,
        StringStruct,
        StringTable,
        VarFileInfo,
        VarStruct,
        VSVersionInfo,
    )

    version_tuple = normalized_version(__version__)
    version_info = VSVersionInfo(
        ffi=FixedFileInfo(filevers=version_tuple, prodvers=version_tuple),
        kids=[
            StringFileInfo(
                [
                    StringTable(
                        '040904B0',
                        [
                            StringStruct('CompanyName', 'Daru'),
                            StringStruct('FileDescription', APP_NAME),
                            StringStruct('FileVersion', __version__),
                            StringStruct('InternalName', APP_EXECUTABLE_NAME),
                            StringStruct(
                                'OriginalFilename',
                                f'{APP_EXECUTABLE_NAME}.exe',
                            ),
                            StringStruct('ProductName', APP_NAME),
                            StringStruct('ProductVersion', __version__),
                        ],
                    )
                ]
            ),
            VarFileInfo([VarStruct('Translation', [1033, 1200])]),
        ],
    )


a = Analysis(
    ['daru_gui.py'],
    pathex=[str(project_root / 'src')],
    binaries=[],
    datas=[],
    hiddenimports=daru_hiddenimports + [
        'ezdxf',
        'deep_translator',
        'deepl',
        'openai',
        'boto3',
        'botocore',
        's3transfer',
        'fitz',
        'pymupdf',
        'pymupdf.table',
        'pdf2image',
        'pytesseract',
        'PyPDF2',
        'lxml',
        'lxml.etree',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_EXECUTABLE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=version_info,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_EXECUTABLE_NAME,
)
