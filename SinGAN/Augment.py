# import albumentations as A
# import SinGAN.functions as functions
# import cv2

# # Declare an augmentation pipeline
# transform = A.Compose([
#     # A.RandomCrop(width=256, height=256),
#     # A.HorizontalFlip(p=0.5),
#     # A.RandomBrightnessContrast(p=0.2),
#     A.Rotate(limit=10, p=0.5), 
#     #A.ShiftScaleRotate
# ])

# def Augment(path):
# # Read an image with OpenCV and convert it to the RGB colorspace
#     print(path)
#     path_ = '\"'+path+'\"'
#     print(path_)
#     image = cv2.imread(path_)
#     print(type(image))
#     #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Augment an image
#     transformed = transform(image=image)
#     transformed_image = transformed["image"]
#     #cv2.imwrite("result.png", transformed_image)
#     return functions.np2torch(transformed_image)

import albumentations as A
import SinGAN.functions as functions
import cv2

# Declare an augmentation pipeline
transform = A.Compose([
    # A.RandomCrop(width=256, height=256),
    # A.HorizontalFlip(p=0.5),
    # A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5), 
    #A.ShiftScaleRotate
])

def Augment(path, opt):
# # Read an image with OpenCV and convert it to the RGB colorspace
    #print(path)
    path_ = '\".'+path+'\"'
    #print(path_)
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    #print(type(image))
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Augment an image
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    #cv2.imwrite("result.png", transformed_image)
    x = functions.np2torch(transformed_image, opt)

    return functions.adjust_scales2image(x, opt)