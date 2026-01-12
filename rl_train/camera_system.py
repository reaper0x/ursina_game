from ursina import *
from panda3d.core import Camera as PandaCamera
from panda3d.core import PerspectiveLens, NodePath
import numpy as np
import config

class GridCameraSystem(Entity):
    def __init__(self, agents):
        super().__init__()
        self.cameras = []
        self.regions = []
        self.targets = []
        base = application.base

        limit = min(config.AGENTS_PER_WINDOW, len(agents))
        viewable = agents[:limit]

        if not viewable: return

        cols = int(np.ceil(np.sqrt(len(viewable))))
        rows = int(np.ceil(len(viewable) / cols))
        
        for i in range(len(viewable)):
            t, r = viewable[i]
            self.targets.append(t)
            
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
            cnp.reparent_to(scene)
            
            dr.set_camera(cnp)
            self.cameras.append(cnp)
            self.regions.append(dr)

    def update(self):
        for i, cam in enumerate(self.cameras):
            if i < len(self.targets) and self.targets[i]:
                target = self.targets[i]
                desired_pos = target.position + Vec3(0, 10, -10)
                cam.set_pos(desired_pos)
                cam.look_at(target.position)

    def cleanup(self):
        base = application.base
        for dr in self.regions:
            base.win.remove_display_region(dr)
        for cam in self.cameras:
            cam.remove_node()
        self.cameras.clear()
        self.regions.clear()
        self.targets.clear()
        destroy(self)