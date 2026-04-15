import cv2
import numpy as np
from ultralytics import YOLO

class SurgicalMonitor:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.3):
        """
        Initializes the YOLOv8 model for surgical monitoring.
        Since we might be using a generic pretrained model for the prototype,
        we'll just detect the objects it knows (e.g. 'scissors', 'bottle').
        """
        self.model = YOLO(model_path)
        # Load the generic COCO model alongside the custom model
        # so it can continue to detect people, phones, etc.
        self.generic_model = YOLO('yolov8n.pt') 
        self.conf_threshold = confidence_threshold

        # Default focus zone margin (can be overridden per frame)
        self.focus_zone_margin = 0.2

        # Classes to ignore from the generic COCO model to avoid mislabeling surgical tools
        self.ignored_generic = ["scissors", "knife", "fork", "spoon"]

    def get_focus_zone(self, frame_width, frame_height, margin=None):
        m = margin if margin is not None else self.focus_zone_margin
        x1 = int(frame_width * m)
        y1 = int(frame_height * m)
        x2 = int(frame_width * (1 - m))
        y2 = int(frame_height * (1 - m))
        return (x1, y1, x2, y2)

    def is_outside_focus(self, bbox, focus_zone):
        """
        Check if an object's bounding box is mostly outside the focus zone.
        bbox format: [x1, y1, x2, y2]
        focus_zone format: (x1, y1, x2, y2)
        """
        bx1, by1, bx2, by2 = bbox
        fx1, fy1, fx2, fy2 = focus_zone

        # Check if ANY part of the bounding box is outside the focus zone
        if bx1 < fx1 or bx2 > fx2 or by1 < fy1 or by2 > fy2:
            return True
        return False

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes."""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        if inter_area == 0:
            return 0.0

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def process_frame(self, frame, custom_conf=None, alert_margin=None, show_focus_zone=True, show_generic=True):
        """
        Analyzes a single frame, runs detection, and draws alerts.
        Returns: (annotated_frame, analytics_dict)
        """
        conf_to_use = custom_conf if custom_conf is not None else self.conf_threshold
        
        # Run BOTH models on the same frame!
        custom_results = self.model(frame, conf=conf_to_use, verbose=False)
        generic_results = self.generic_model(frame, conf=0.15, verbose=False)

        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        focus_zone = self.get_focus_zone(w, h, margin=alert_margin)
        
        # Draw focus zone if toggled on
        if show_focus_zone:
            cv2.rectangle(annotated_frame, (focus_zone[0], focus_zone[1]), (focus_zone[2], focus_zone[3]), (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Primary Focus Zone", (focus_zone[0], focus_zone[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Collect generic boxes that shouldn't be ignored for IoU filtering
        valid_generic_boxes = []
        for r in generic_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = self.generic_model.names[cls_id]
                if class_name not in self.ignored_generic:
                    gx1, gy1, gx2, gy2 = box.xyxy[0].cpu().numpy()
                    valid_generic_boxes.append((int(gx1), int(gy1), int(gx2), int(gy2)))

        analytics = {
            "instrument_count": 0,
            "generic_count": 0,
            "outside_alerts": 0,
            "total_confidence": 0.0,
            "avg_confidence": 0.0
        }

        # Helper function to draw boxes
        def draw_boxes(results_list, annotated_img, fz, is_generic=False):
            for r in results_list:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Confidence and class ID
                    conf = box.conf[0]
                    cls_id = int(box.cls[0])
                    
                    if is_generic:
                        class_name = self.generic_model.names[cls_id]
                        
                        # Filter out specific generic classes that overlap with surgical tools
                        if class_name in self.ignored_generic:
                            continue
                            
                        # Change color to Purple for generic objects to distinguish them
                        color = (255, 0, 255) 
                        bg_color = (255, 0, 255)
                        text_color = (255, 255, 255)
                    else:
                        class_name = self.model.names[cls_id]

                        box_area = (x2 - x1) * (y2 - y1)
                        frame_area = annotated_img.shape[0] * annotated_img.shape[1]
                        
                        # Hard constraint: A surgical tool should not take up more than 30% of the entire frame area
                        if box_area > frame_area * 0.30:
                            continue
                        
                        # Cross-model NMS: Ignore if it overlaps heavily with a generic object (like a person)
                        suppress = False
                        for g_box in valid_generic_boxes:
                            iou = self.calculate_iou((x1, y1, x2, y2), g_box)
                            if iou > 0.25:  # Lowered threshold to reliably catch the overlap
                                suppress = True
                                break
                        if suppress:
                            continue
                            
                        color = (255, 0, 0) # Blue for surgical tools
                        bg_color = color
                        text_color = (255, 255, 255)
                    
                    if not is_generic:
                        analytics["instrument_count"] += 1
                        analytics["total_confidence"] += conf
                        
                        if self.is_outside_focus([x1, y1, x2, y2], fz):
                            analytics["outside_alerts"] += 1
                            color = (0, 0, 255) # Red for alert bounding box
                            bg_color = (0, 255, 255) # Yellow background for text
                            text_color = (0, 0, 0) # Black text
                            text = f"ALERT: {class_name} outside focus!"
                        else:
                            text = f"{class_name} {conf:.2f}"
                    else:
                        analytics["generic_count"] += 1
                        text = f"{class_name} {conf:.2f}"

                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    
                    # Ensure text doesn't go off top of screen
                    text_y = max(y1, 25)
                    
                    # Background rectangle
                    cv2.rectangle(annotated_img, (x1, text_y - 25), (x1 + text_size[0] + 5, text_y), bg_color, -1)
                    
                    # Box and Text
                    line_thick = 3 if not is_generic else 2
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, line_thick)
                    cv2.putText(annotated_img, text, (x1 + 2, text_y - 7), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        # Draw generic boxes first so surgical tools are always drawn on top
        if show_generic:
            draw_boxes(generic_results, annotated_frame, focus_zone, is_generic=True)
        draw_boxes(custom_results, annotated_frame, focus_zone, is_generic=False)

        if analytics["instrument_count"] > 0:
            analytics["avg_confidence"] = analytics["total_confidence"] / analytics["instrument_count"]

        return annotated_frame, analytics
