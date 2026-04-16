from ultralytics import YOLO

def main():
    model = YOLO('best.pt')
    results = model('../datasets/surgical_data/test/images/104_jpg.rf.8762744b7ea35272513266f4ed6a2d3e.jpg', conf=0.01)
    print("\n--- RESULTS ---")
    for r in results:
        print("Boxes:", len(r.boxes))
        if len(r.boxes) > 0:
            print("Cls:", r.boxes.cls)
            print("Conf:", r.boxes.conf)
            print("Names:", [model.names[int(c)] for c in r.boxes.cls])

if __name__ == "__main__":
    main()
