import cv2
from yolov8_surgical_monitor import SurgicalMonitor

def main():
    m = SurgicalMonitor('best.pt', 0.15)
    img = cv2.imread('../datasets/surgical_data/test/images/104_jpg.rf.8762744b7ea35272513266f4ed6a2d3e.jpg')
    res, stats = m.process_frame(img, 0.15, 0.2)
    print("Stats:", stats)

if __name__ == "__main__":
    main()
