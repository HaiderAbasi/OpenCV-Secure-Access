# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_data_files


block_cipher = None


models = [
('./models/authorized_personnels-live_embeddings-face_enc', './models'),
]

datas = []
datas += collect_data_files('face_recognition')
datas += collect_data_files('face_recognition_models')
datas += copy_metadata('face_recognition')
datas += copy_metadata('face_recognition_models')


a = Analysis(
    ['core.py'],
    pathex=[],
    binaries=models,
    datas=datas,
    hiddenimports=[],
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
    name='core',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='core',
)
