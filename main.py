import cv2 as cv
import torch
from tensorflow import keras
import tensorflow as tf
import numpy as np
from torchvision import transforms
from visualization import visual

class QuadDetection():
    def __init__(self, segmentation: bool):
        self.segmentation = segmentation
        self.load_model()
        self.quad_detection()

    def load_model(self) -> None:
        if self.segmentation:
            self.model = keras.models.load_model('models/NeuralNet')
        else:
            self.model = torch.load('models/quadro_detector_frcnn_fast.pth')

    def quad_detection(self):
        vid_capture = cv.VideoCapture("123.mkv")
        if self.segmentation:
            while (vid_capture.isOpened()):
                ret, frame = vid_capture.read()
                if ret == True:
                    frame = cv.resize(frame, (224, 224))
                    image = tf.keras.preprocessing.image.img_to_array(frame).reshape(1, 224, 224, 3)
                    segmentation_mask = self.model(image)
                    segmentation_mask = np.asarray(segmentation_mask, dtype=np.float32)
                    mask = np.copy(segmentation_mask)
                    for i in range(224):
                        for j in range(224):
                            if mask[0][i][j] > 0.2:
                                mask[0][i][j] = 255
                            else:
                                mask[0][i][j] = 0
                    segmentation_mask = np.asarray(mask, dtype=np.uint8)
                    contours, hierarchy = cv.findContours(segmentation_mask.reshape((224, 224, 1)), cv.RETR_TREE,
                                                          cv.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        area = cv.contourArea(contour)
                        if area > 200:
                            # Найденные координаты описывающего объект квадрата
                            x, y, w, h = cv.boundingRect(contour)
                            print(x,y,w,h)
                else:
                    break
        else:
            while (vid_capture.isOpened()):
                ret, frame = vid_capture.read()
                if ret == True:
                    frame = cv.resize(frame, (224, 224))
                    image = transforms.ToTensor()(frame).unsqueeze_(0)
                    output = self.model(image)
                    bboxes = output[0]['boxes'].detach().numpy()
                    image1 = visual(output[0]['boxes'].detach().numpy(), frame, output[0]['scores'].detach().numpy(),
                                   output[0]['labels'].detach().numpy())
                    cv.imshow('net', image1)
                    print(bboxes)
                    key = cv.waitKey(20)
                    if key == ord('q'):
                        break
                else:
                    break
        vid_capture.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Параметры для запуска приложения по обнаружению объектов')
    parser.add_argument("-s","--segmentation",type=bool, default=False)

    args = parser.parse_args()
    detection = QuadDetection(args.segmentation)



