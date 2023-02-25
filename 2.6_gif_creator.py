import glob
import contextlib
from PIL import Image


filename_starter = "Plots/Task2.6A_truncation_"
filename_middle = ["1_MINUTE_(A)","1_MINUTE_(B)","2_MINUTES_(A)", "2_MINUTES_(B)", "4_MINUTES_(A)", "4_MINUTES_(B)", "6_MINUTES", "8_MINUTES", "16_MINUTES"]
filename_ender = "_n="
iter = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
#, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49
filenames = []
for i in range(len(filename_middle)):
    filenames = []
    for j in range(len(iter)):
        filename_transient = filename_starter + filename_middle[i] + filename_ender + str(iter[j]) + ".png"
        filenames.append(filename_transient)
        
        fp_out = "Plots/Task2.7A_truncation_" + filename_middle[i] + ".gif"

        with contextlib.ExitStack() as stack:
            imgs = (stack.enter_context(Image.open(f))
                    for f in filenames)
            img = next(imgs)
            img.save(fp=fp_out, format='GIF', append_images=imgs,
                    save_all=True, duration=200, loop=0)

