from ursina import *
from panda3d.core import Camera as PandaCamera
from panda3d.core import PerspectiveLens, NodePath
import numpy as np
import config

class GridCameraSystem:
    def __init__(self, agents):
        self.cameras = []
        self.regions = []
        base = application.base

        limit = min(config.AGENTS_PER_WINDOW, len(agents))
        viewable = agents[:limit]

        if not viewable: return

        cols = int(np.ceil(np.sqrt(len(viewable))))
        rows = int(np.ceil(len(viewable) / cols))
        
        for i in range(len(viewable)):
            t, r = viewable[i]
            col = i % cols
            row = i // cols
            
            l = col / cols
            r_edge = (col + 1) / cols
            b = 1.0 - ((row + 1) / rows)
            t_edge = 1.0 - (row / rows)
            
            dr = base.win.makeDisplayRegion(l, r_edge, b, t_edge)
            dr.set_clear_color_active(True)
            dr.set_clear_color((0.1, 0.1, 0.1, 1))
            
            cam = PandaCamera(f'grid_cam_{i}')
            cam.set_lens(PerspectiveLens())
            cam.get_lens().set_fov(90)
            cam.set_scene(scene)
            
            cnp = NodePath(cam)
            cnp.reparent_to(t)
            cnp.set_pos(0, 6, -9)
            cnp.look_at(t)
            
            dr.set_camera(cnp)
            self.cameras.append(cnp)
            self.regions.append(dr)

    def cleanup(self):
        base = application.base
        for dr in self.regions:
            base.win.remove_display_region(dr)
        for cam in self.cameras:
            cam.remove_node()
        self.cameras.clear()
        self.regions.clear()