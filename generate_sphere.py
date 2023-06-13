import pyvista as pv
import numpy as np
import json
import math

sphere = pv.Sphere(radius=1)
plotter = pv.Plotter()
plotter.add_mesh(sphere, color="white")

positions = [
    [0, 5 / math.sqrt(2), -5 / math.sqrt(2)],  # front view
    [0, 5, 0],  # top view
    [5, 0, 0],  # right view
    [-5 / math.sqrt(2), 0, 5 / math.sqrt(2)],  # back view
    [0, -5, 0],  # bottom view
    [-5, 0, 0],  # left view
]


def data_to_json(dict):
    json_object = json.dumps(dict, indent=4)
    with open("sphere_data.json", "a") as outfile:
        outfile.write(json_object + ",\n")


def arr_to_4by4(arr):
    return np.array(
        [
            [arr[0], arr[1], arr[2], arr[3]],
            [arr[4], arr[5], arr[6], arr[7]],
            [arr[8], arr[9], arr[10], arr[11]],
            [arr[12], arr[13], arr[14], arr[15]],
        ]
    )


for i, pos in enumerate(positions):
    plotter.camera.position = pos
    camera = plotter.camera
    trans_matrix = camera.GetViewTransformMatrix()
    print(trans_matrix)
    plotter.show(
        title=f"View {i+1}",
        screenshot=f"./generated_views/view{i+1}.png",
        interactive_update=True,
    )
    trans_matrix_list = []
    for i in range(4):
        for j in range(4):
            trans_matrix_list.append(trans_matrix.GetElement(i, j))
    trans_matrix_list = arr_to_4by4(trans_matrix_list)
    data_to_json(trans_matrix_list.tolist())
