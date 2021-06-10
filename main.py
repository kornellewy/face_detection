import os
from pathlib import Path
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
import torch

# WAZNE!!!
# pip install facenet-pytorch


class FaceDetecor2(object):
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = MTCNN(keep_all=True, device=self.device)

    def detect_from_numpy(self, numpy_img):
        return self._detect_from_numpy(numpy_img)

    def _detect_from_numpy(self, numpy_img):
        image = Image.fromarray(cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB))
        boxes, _ = self.model.detect(image)
        image_draw = image.copy()
        draw = ImageDraw.Draw(image_draw)
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
        return boxes, image_draw
        

if __name__ == '__main__':
    image_path = 'test.jpg'
    kjn = FaceDetecor2()
    img = cv2.imread(image_path)
    bboxes, img = kjn.detect_from_numpy(img)
    img.show()
    