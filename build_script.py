import PyInstaller.__main__
import os

PyInstaller.__main__.run([
    'main.py',                         
    '--name=TagGame3D',                
    '--onefile',                       
    '--windowed',                      
    '--clean',                         
    '--noconfirm',                     
    
    '--hidden-import=ursina',
    '--hidden-import=ursina.prefabs',
    '--hidden-import=panda3d',
    '--hidden-import=screeninfo',
    
    '--collect-all=ursina',
    '--collect-all=panda3d',
])
