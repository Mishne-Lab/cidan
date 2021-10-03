import os

import fire
from tifffile import tifffile
import numpy as np
import matplotlib.pyplot as plt

def apply_possion(input_file, output_file, multiplier=.5, number=1):
    rng = np.random.default_rng()
    image = tifffile.imread(input_file)
    image = image.astype("float64")
    if np.isclose(np.mean(image[:50]),2**15, 2000):
        image = image - (2 ** 15)
        image[image < 0] = 0
    image = image[:,10:245,10:245]
    for x in range(number):


        std = np.std(image,axis=0)
        # std = std[:, 10:245, 10:245]
        # plt.imshow(std)

        # plt.imshow(np.max(rng.poisson(std, (500,235, 235)).astype("uint16"), axis=0))
        # plt.show()
        noise = rng.poisson(std*multiplier,image.shape)

        # noise = noise.transpose([2,0,1])
        image_w_noise = image+noise
        if number!=1:
            output_file_n = output_file[:-4]+"_"+str(x)+"_5.tif"
        else:
            output_file_n = output_file
        with open(output_file_n, "wb") as f:
            tifffile.imsave(f, image_w_noise.astype("uint16"))
            print(output_file_n)

if __name__ == '__main__':
    fire.Fire(apply_possion)