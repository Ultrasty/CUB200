import glob
import numpy as np
import shutil

images_path = glob.glob("CT/*/cap033/*.jpg")
all_labels_name = [img_p.split('\\')[0] + "\\" + img_p.split('\\')[1] + "\\" + img_p.split('\\')[2] for img_p in
                   images_path]

labelName = np.unique(all_labels_name)

for i in labelName:
    path = glob.glob(i + "/*.jpg")
    List = []
    for i in range(len(path)):
        List.append(i)

    ListBefore = []
    for i in List:
        ListBefore.append(i)

    for index, i in enumerate(ListBefore):
        ListBefore[index]="CT"+ str(ListBefore[index]).rjust(4,'0')+".jpg"

    for index, i in enumerate(List):
        List[index] = str(List[index]+1)

    List.sort()
    for index, i in enumerate(List):
        List[index] = "2-CT"+List[index].rjust(4,'0')+".jpg"


    for index, i in enumerate(List):
        shutil.move(labelName[0]+"\\"+ListBefore[index],labelName[0]+"\\"+List[index])

print(labelName[0])