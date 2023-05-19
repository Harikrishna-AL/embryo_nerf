import pyvista as pv
import numpy as np
import json

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
def data_to_json(dict):
    json_object = json.dumps(dict, indent=4)
    with open("sphere_data.json", "a") as outfile:
        outfile.write(json_object)
# while True:
for i, pos in enumerate(positions):
    # plotter.camera_position = pos
    plotter.camera.position = pos
    # plotter.camera_set = True
    camera = plotter.camera
    trans_matrix = camera.GetViewTransformMatrix()
    # trans_matrix = plotter.get_view_matrix()
    plotter.show(title=f"View {i+1}",screenshot=f"./generated_views/view{i+1}.png",interactive_update=True)
    # print(plotter.camera.position)
    # print(np.array(trans_matrix))
    trans_matrix_list = []
    for i in range(4):
        for j in range(4):
            trans_matrix_list.append(trans_matrix.GetElement(i, j))
    # trans_matrix_np = np.array(trans_matrix.GetElement(0,0))
    data_to_json(trans_matrix_list)

# dictionary = {
#     "name": "sathiyajith",
#     "rollno": 56,
#     "cgpa": 8.6,
#     "phonenumber": "9976770500"
# }
