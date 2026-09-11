from PIL import Image
import numpy as np

img = Image.open("dataset_orientado/dataset_train/cancer/4177/D_4177_1.LEFT_CC.jpg")
arr = np.array(img)

print(img.mode)
print(arr.shape)

if arr.ndim == 3 and arr.shape[2] == 3:
    same = np.all(arr[:,:,0] == arr[:,:,1]) and np.all(arr[:,:,1] == arr[:,:,2])
    print("RGB com canais iguais?" , same)