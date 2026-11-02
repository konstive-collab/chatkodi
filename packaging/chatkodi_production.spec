# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs, collect_submodules


ROOT = Path(SPECPATH).resolve().parent
if ROOT.name.lower() == "packaging":
    ROOT = ROOT.parent
DIST_NAME = "ChatKodi"


def _collect_package(package):
    try:
        return collect_all(package)
    except Exception:
        return [], [], []


def _runtime_hidden_import(name):
    lowered = name.lower()
    excluded_tokens = (
        ".tests",
        ".test_",
        ".testing",
        ".test_utils",
        ".examples",
        ".example",
        ".benchmarks",
        ".benchmark",
    )
    return not any(token in lowered for token in excluded_tokens)


datas = [
    (str(ROOT / "chatkodi_config.json"), "."),
    (str(ROOT / "VERSION.txt"), "."),
    (str(ROOT / "manuals"), "manuals"),
    (str(ROOT / "assets" / "fonts"), "assets/fonts"),  # Onest / Unbounded (SIL OFL)
]

binaries = []
hiddenimports = []

# Models run in llama.cpp (llama-server); no torch/transformers in the bundle.
for package in (
    "PyQt5",
    "huggingface_hub",
    "numpy",
    "docx",
    "PyPDF2",
    "openpyxl",
    "pptx",
    "odf",
    "bs4",
    "striprtf",
    "xlrd",
    "pymorphy3",
    "pymorphy3_dicts_ru",
    "PIL",  # photos for the vision model
    "pypdfium2",  # scanned PDF pages -> images
    "pypdfium2_raw",
):
    pkg_datas, pkg_binaries, pkg_hiddenimports = _collect_package(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += [name for name in pkg_hiddenimports if _runtime_hidden_import(name)]

for package in ("core", "gui", "tools", "plugins"):
    hiddenimports += [name for name in collect_submodules(package) if _runtime_hidden_import(name)]

excludes = [
    "tkinter",
    "matplotlib",
    "pytest",
    "IPython",
    "tensorboard",
    "dask",
    "numpy.tests",
    "torch",
    "transformers",
    "sentence_transformers",
    "accelerate",
    "tokenizers",
    "safetensors",
    "sklearn",
    "scipy",
]

app_analysis = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
app_pyz = PYZ(app_analysis.pure)
app_exe = EXE(
    app_pyz,
    app_analysis.scripts,
    [],
    exclude_binaries=True,
    name="ChatKodiApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

launcher_analysis = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
launcher_pyz = PYZ(launcher_analysis.pure)
launcher_exe = EXE(
    launcher_pyz,
    launcher_analysis.scripts,
    [],
    exclude_binaries=True,
    name="ChatKodiLauncher",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

all_binaries = app_analysis.binaries + launcher_analysis.binaries
all_datas = app_analysis.datas + launcher_analysis.datas
all_zipfiles = app_analysis.zipfiles + launcher_analysis.zipfiles

coll = COLLECT(
    app_exe,
    launcher_exe,
    all_binaries,
    all_zipfiles,
    all_datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    contents_directory=".",
    name=DIST_NAME,
)
