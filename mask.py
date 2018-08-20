import numpy as np
# import matplotlib.pyplot as plt

class Mask_obj:
    def __init__(self, shape=(256,256,3), top=64, left=64, width=128, height=128):
        self.shape = shape
        self.top = top
        self.left = left
        self.width = width
        self.height = height
        self.bbox = (top, left, width, height)
        self.image = self.generate_image()

    def generate_image(self):
        top = self.top
        left = self.left
        width = self.width
        height = self.height
        image = np.zeros(self.shape).astype('uint8')
        image[top:top + height, left:left + width, :] = 255
        return image



def generate_submask_objs(mask_objs):
    submask_objs = []

    for obj in mask_objs:
        shape = obj.shape
        top, left, width, height = obj.bbox
        w, h = width // 2, height // 2

        for i in range(2):
            for j in range(2):
                submask_obj = Mask_obj(shape, top + i * h, left + j * w, w, h)
                submask_objs.append(submask_obj)

    return submask_objs

mask_obj = Mask_obj()

submask_objs = generate_submask_objs([mask_obj])

# for i in range(4):
#     plt.imshow(submask_objs[i].img)
#
# print(mask_obj.width)
