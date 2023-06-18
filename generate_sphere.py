import pyvista as pv
import numpy as np
import json
import math

sphere = pv.Sphere(radius=1)
plotter = pv.Plotter()
plotter.add_mesh(sphere, color="white")


def circle_xy(init_angle, end_angle, radius, num_samples):
    angles = np.linspace(init_angle, end_angle, num_samples)
    x = np.cos(angles) * radius
    y = np.sin(angles) * radius
    pos = np.array([x, y, np.zeros_like(x)])
    pos = pos.transpose()
    return pos


def data_to_json(dict, idx, last_idx):
    json_object = json.dumps(dict, indent=4)
    if idx != last_idx:
        with open("sphere_data.json", "a") as outfile:
            outfile.write('"' + str(idx + 1) + '"' + ":" + json_object + ",\n")
    else:
        with open("sphere_data.json", "a") as outfile:
            outfile.write('"' + str(idx + 1) + '"' + ":" + json_object + "\n")


def arr_to_4by4(arr):
    return np.array(
        [
            [arr[0], arr[1], arr[2], arr[3]],
            [arr[4], arr[5], arr[6], arr[7]],
            [arr[8], arr[9], arr[10], arr[11]],
            [arr[12], arr[13], arr[14], arr[15]],
        ]
    )


positions = circle_xy(0, 2 * np.pi, 5, 100)

with open("sphere_data.json", "w") as outfile:
    outfile.write("{\n")

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
    for k in range(4):
        for j in range(4):
            trans_matrix_list.append(trans_matrix.GetElement(k, j))
    trans_matrix_list = arr_to_4by4(trans_matrix_list)
    data_to_json(trans_matrix_list.tolist(), i, len(positions) - 1)

with open("sphere_data.json", "a") as outfile:
    outfile.write("}")
