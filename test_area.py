import cv2
from ultralytics import YOLO

m = YOLO('best.pt')
res = m('../datasets/surgical_data/test/images/104_jpg.rf.8762744b7ea35272513266f4ed6a2d3e.jpg', conf=0.15)
img=cv2.imread('../datasets/surgical_data/test/images/104_jpg.rf.8762744b7ea35272513266f4ed6a2d3e.jpg')
fa=img.shape[0]*img.shape[1]
print('FA',fa)
b=res[0].boxes[0]
x1,y1,x2,y2=b.xyxy[0]
print('BA:', (x2-x1)*(y2-y1))
print('BAratio', ((x2-x1)*(y2-y1))/fa)
