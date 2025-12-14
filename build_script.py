import PyInstaller.__main__
import os

# Define the build command programmatically
PyInstaller.__main__.run([
    'main.py',                         # Your main file
    '--name=TagGame3D',                # Name of the exe
    '--onefile',                       # Single file
    '--windowed',                      # No black console (change to --console to debug)
    '--clean',                         # Clean cache before building
    '--noconfirm',                     # Don't ask for confirmation
    
    # FORCE IMPORT URSINA AND PANDA3D
    '--hidden-import=ursina',
    '--hidden-import=ursina.prefabs',
    '--hidden-import=panda3d',
    '--hidden-import=screeninfo',
    
    # COLLECT ALL URSINA DATA
    '--collect-all=ursina',
    '--collect-all=panda3d',
])
