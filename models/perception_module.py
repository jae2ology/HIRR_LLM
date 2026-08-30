import cv2
import numpy as np
from ultralytics import YOLO

USE_ISAAC = False

if USE_ISAAC:
    from omni.isaac.kit import SimulationApp
    from omni.isaac.sensor import Camera


class PerceptionModule:
    def __init__(self, use=USE_ISAAC):
        self.use_sim = use
        self.yolo_model = YOLO('yolov8n.pt')

        if self.use_sim:
            self.camera = Camera(prim_path="/World/Stretch/head_camera", resolution=(640, 480))
            self.camera.initialize()
            self.camera.add_distance_to_image_plane_to_frame()

    def get_rgbd(self):
        """ returns (bgr_image, depth_array)"""
        if self.use_sim:
            rgba = self.camera.get_rgba()
            depth = self.camera.get_current_frame().get("distance_to_image_plane")
            if rgba is None:
                return None, None

            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            return bgr, depth
        else:
            dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_depth = np.full((480, 640), 1.5, dtype=np.float32)
            return dummy_bgr, dummy_depth

    def get_detected_objects(self):
        bgr, depth = self.get_rgbd()
        if bgr is None:
            return []

        if not self.use_sim:
            return ["toothbrush"]  # for simulated results

        results = self.yolo_model(bgr, verbose=False)
        detected_labels = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.yolo_model.names[cls_id]
                detected_labels.append(label)

        return detected_labels