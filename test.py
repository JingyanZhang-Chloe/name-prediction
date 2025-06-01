import data
import numpy as np

classes = {}
_sum = 0
for x, y in data.val_dataset:
    for z in y:
        z = int(z)
        if not (z in classes):
            classes[z] = 0
        _sum += 1
        classes[z] += 1

for y in classes:
    classes[y] = classes[y]

print(sorted(classes.items(), key=lambda p: p[1]))
print(sorted(classes.keys()))
