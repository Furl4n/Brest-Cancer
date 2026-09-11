import pywt
import numpy as np
from PIL import Image
import os
from torchvision import transforms


class WaveletMosaicTransform:
    def __init__(self, wavelet='db4', output_dir='imagens_artigo'):
        self.wavelet = wavelet
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def norm(self, c):
        c_min, c_max = np.percentile(c, 1), np.percentile(c, 99)
        c = np.clip(c, c_min, c_max)

        if c_max > c_min:
            c = (c - c_min) / (c_max - c_min)
        else:
            c = np.zeros_like(c)

        return (c * 255).astype(np.uint8)

    def save_array(self, arr, filename):
        Image.fromarray(arr, mode='L').resize((224, 224)).save(os.path.join(self.output_dir, filename))

    def __call__(self, img):
        arr = np.array(img.convert("L"), dtype=np.float32)

        LL, (LH, HL, HH) = pywt.dwt2(arr, self.wavelet, mode='symmetric')

        # normaliza uma única vez
        LLn = self.norm(LL)
        LHn = self.norm(LH)
        HLn = self.norm(HL)
        HHn = self.norm(HH)

        # salva exatamente os mesmos arrays usados no mosaico
        self.save_array(LLn, 'LL.png')
        self.save_array(LHn, 'LH.png')
        self.save_array(HLn, 'HL.png')
        self.save_array(HHn, 'HH.png')

        top = np.hstack((LLn, HLn))
        bottom = np.hstack((LHn, HHn))
        mosaic = np.vstack((top, bottom))

        return Image.fromarray(mosaic, mode='L')


wavelet_transform = WaveletMosaicTransform()

test_pipeline = transforms.Compose([
    wavelet_transform,
    transforms.Resize((224, 224))
])

img_teste = test_pipeline(
    Image.open('dataset_orientado/dataset_train/benigno/0217/C_0217_1.LEFT_CC.jpg')
)

Image.open('dataset_orientado/dataset_train/benigno/0217/C_0217_1.LEFT_CC.jpg').resize((224, 224)).save("imagens_artigo/Imagem_real.png");

img_teste.save('imagens_artigo/mosaico_completo.png')