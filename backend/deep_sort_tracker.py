import os
import sys
import numpy as np

# Add path to 'tracker' so deep_sort modules can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tracker")))

from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.application_util.preprocessing import non_max_suppression
from deep_sort.deep.reid_encoder import FeatureExtractor


class DeepSort:
    def __init__(self, model_path=None):
        print("[INFO] Initializing DeepSort with PyTorch-based encoder...")

        if model_path is None:
            model_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__),
                "..", "tracker", "deep_sort", "deep", "checkpoint", "resnet18.pth"
            ))

        print("[INFO] Using DeepSort reID model:", model_path)

        self.encoder = FeatureExtractor(model_path)

        max_cosine_distance = 0.4
        nn_budget = None
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def update_tracks(self, detections, frame):
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            return []

        # Extract bounding boxes and scores
        bboxes = np.array([det[:4] for det in detections])
        scores = np.array([det[4] for det in detections])

        # Extract appearance features from crops
        features = self.encoder.extract_features(frame, bboxes)
        if len(features) == 0:
            self.tracker.predict()
            self.tracker.update([])
            return []

        dets = [Detection(bbox, score, feature) for bbox, score, feature in zip(bboxes, scores, features)]

        # Apply non-max suppression
        boxes = np.array([d.tlwh for d in dets])
        confidences = np.array([d.confidence for d in dets])
        indices = non_max_suppression(boxes, 0.7, confidences)
        dets = [dets[i] for i in indices]

        # Update tracker
        self.tracker.predict()
        self.tracker.update(dets)

        # Collect confirmed tracks
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            outputs.append(TrackedObject(track.track_id, bbox))

        return outputs


class TrackedObject:
    def __init__(self, track_id, tlwh):
        self.track_id = track_id
        self.tlwh = tlwh

    def to_ltrb(self):
        x, y, w, h = self.tlwh
        return x, y, x + w, y + h
