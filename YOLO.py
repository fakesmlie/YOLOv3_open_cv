import cv2
import numpy as np

import matplotlib.pyplot as plt


def look_img(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

with open('coco.names') as f:
    classes = f.read().splitlines()

img = cv2.imread('kite.jpg')

height, width, _ = img.shape
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

net.setInput(blob)

# layersNames = net.getLayerNames()
# output_layers_names = [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
output_layers_names = ['yolo_82', 'yolo_94', 'yolo_106']

prediction = net.forward(output_layers_names)

boxes = []  # 存放预测框坐标
objectness = []  # 存放置行度
class_probs = []  # 存放类别概率
class_ids = []  # 存放预测框类别索引号
class_names = []  # 存放预测框类别名称

for scale in prediction:  # 遍历三种尺度
    for bbox in scale:  # 遍历每一个预测框
        obj = bbox[4]
        class_scores = bbox[5:]
        class_id = np.argmax(class_scores)
        class_name = classes[class_id]
        class_prob = class_scores[class_id]

        center_x = int(bbox[0] * width)
        center_y = int(bbox[1] * height)
        w = int(bbox[2] * width)
        h = int(bbox[3] * height)

        x = int(center_x - w / 2)
        y = int(center_y - h / 2)

        boxes.append([x, y, w, h])
        objectness.append(float(obj))
        class_ids.append(class_id)
        class_names.append(class_name)
        class_probs.append(class_prob)

confidences = np.array(class_probs) * np.array(objectness)

conf_thres = 0.1
nms_thres = 0.4

indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, nms_thres)

colors = np.random.uniform(0, 255, size=(len(boxes), 3))

for i in indexes.flatten():
    x, y, w, h = boxes[i]
    confidence = str(round(confidences[i], 2))
    color = colors[i % len(colors)]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 8)
    string = '{}{}'.format(class_names[i], confidence)
    cv2.putText(img, string, (x, y+20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 5)

look_img(img)

cv2.imwrite('result-test.jpg', img)
