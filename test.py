import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

import albumentations as A

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)

    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.95,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )

    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = f'{category_id_to_name[category_id]}'
        img = visualize_bbox(img, bbox, class_name)
    cv2.imshow("image", img)
    cv2.imwrite("10.jpg", img)

    cv2.waitKey(0)


def visualizeImg(image):
    img = image.copy()
    #plt.figure(figsize=(12, 12))
    #plt.axis('off')
    #plt.imshow(img)
    cv2.imshow("image", img)
    cv2.waitKey(0)

image = cv2.imread('/Users/david/Downloads/241696775_361488712316215_3058648716608776870_n.jpg')
image2 = cv2.imread('/Users/david/Downloads/241696775_361488712316215_3058648716608776870_n.jpg')

print('aaa', image.shape[:2])

bboxes = [[306.7, 130.84, 350.8, 230.84],
            [510, 220, 221, 239]

          #[386.7, 100.84, 26.8, 46.84]
           ]
category_ids = [17, 18

                ]

# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image
category_id_to_name = {17: 'book', 18: 'face'
                       }
#category_id_to_name = {17: 'cat', 18: 'dog'}

transform = A.Compose([
    #A.RandomGridShuffle(always_apply=True),
    A.UnpropShuffle(p=1.0),
    #A.CropNonEmptyMaskIfExists(always_apply=True, height=300, width=300),
    #A.HorizontalFlip(p=1.0 ),
],  bbox_params =A.BboxParams(format = 'coco', label_fields = ['category_ids']),
)

random.seed(7)
transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)


visualize(
image2,
    bboxes,
    category_ids,
    category_id_to_name,
)


visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)

#transformed = transform(image=image)

#cv2.imshow("image", transformed['image'])
#cv2.waitKey(0)