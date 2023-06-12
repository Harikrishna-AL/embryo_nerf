import json

file = open("./sphere_data.json")
data = json.load(file)
print(data["2"])
