import shutil

data = open('Slice_level_label.csv')
i = 0
list = []
for each_line in data:
    if i > 0:
        list.append(each_line.split(',')[1:200])
    i = i + 1

print(list)

i = 0
count = 0
while i < 80:
    k = 0
    if i < 55:
        for j in list[i]:
            if 20 < k:
                if j == '1':
                    try:
                        shutil.copyfile('./Covid-19/P' + str(1000 + i + 1)[1:] + '/2-CT' + str(10000 + k)[1:] + '.png',
                                        './theCovid-19/' + str(count) + '.png')
                    except Exception:
                        pass
                if j == '0':
                    try:
                        shutil.copyfile('./Covid-19/P' + str(1000 + i + 1)[1:] + '/2-CT' + str(10000 + k)[1:] + '.png',
                                        './theNormal/' + str(count) + '.png')
                    except Exception:
                        pass
            k = k + 1
            count = count + 1
    else:
        for j in list[i]:
            if 20 < k:
                if j == '1':
                    try:
                        shutil.copyfile('./Cap/cap' + str(1000 + i - 54)[1:] + '/2-CT' + str(10000 + k)[1:] + '.png',
                                        './theCap/' + str(count) + '.png')
                    except Exception:
                        pass
                if j == '0':
                    try:
                        shutil.copyfile('./Cap/cap' + str(1000 + i - 54)[1:] + '/2-CT' + str(10000 + k)[1:] + '.png',
                                        './theNormal/' + str(count) + '.png')
                    except Exception:
                        pass
            k = k + 1
            count = count + 1

    i = i + 1
