import cv2 as cv
from tensorflow import keras
import tensorflow as tf
import numpy as np

model = keras.models.load_model('NeuralNet')
vid_capture = cv.VideoCapture(0)
frame_size = (224,224)
while (vid_capture.isOpened()):
    ret, frame = vid_capture.read()
    if ret == True:
        frame = cv.resize(frame,(224,224))
        image = tf.keras.preprocessing.image.img_to_array(frame).reshape(1, 224, 224, 3)
        segmentation_mask = model(image)
        segmentation_mask = np.asarray(segmentation_mask, dtype=np.float32)
        mask = np.copy(segmentation_mask)
        for i in range(224):
            for j in range(224):
                if mask[0][i][j]>0.8:
                    print(mask[0][i][j])
                    mask[0][i][j] = 255
                else:
                    mask[0][i][j] = 0
        segmentation_mask = np.asarray(mask, dtype=np.uint8)
        contours, hierarchy = cv.findContours(segmentation_mask.reshape((224,224,1)), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 200:
                # Найденные координаты описывающего объект квадрата
                x, y, w, h = cv.boundingRect(contour)
                print(x,y,w,h)
        key = cv.waitKey(20)
        if key == ord('q'):
            break
    else:
        break
vid_capture.release()
cv.destroyAllWindows()


