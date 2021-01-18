import glob
import numpy as np
import shutil
import os

images_path = glob.glob("*/*/*.png")
all_labels_name = [img_p.split('\\')[0] + "\\" + img_p.split('\\')[1] for img_p in
                   images_path]

labelName = np.unique(all_labels_name)

for j in labelName:

    path = glob.glob(j + "/*.png")
    List = []
    for i in range(len(path)):
        List.append(i)

    ListBefore = []
    for i in List:
        ListBefore.append(i)

    for index, i in enumerate(ListBefore):
        ListBefore[index] = "2-CT" + str(ListBefore[index]).rjust(4, '0') + ".png"

    for index, i in enumerate(path):
        if index < 30 or index > len(path) - 50:
            os.remove(path[index])

print(labelName[1])
