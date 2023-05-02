import pyvista as pv
import numpy as np

sphere = pv.Sphere(radius=1)
plotter = pv.Plotter()
plotter.add_mesh(sphere, color="white")

positions = [
    [0, 0, 5],    # front view
    [0, 5, 0],    # top view
    [5, 0, 0],    # right view
    [0, 0, -5],   # back view
    [0, -5, 0],   # bottom view
    [-5, 0, 0],   # left view
]

for i, pos in enumerate(positions):
    plotter.camera_position = pos
    plotter.show(title=f"View {i+1}",screenshot=f"./generated_views/view{i+1}.png",interactive_update=True)


