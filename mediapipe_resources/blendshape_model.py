"""Runs MediaPipe's face blendshapes model directly on a set of face landmarks.

Mirrors FaceBlendshapesGraph in
mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc:
  1. take the 146-landmark subset (kLandmarksSubsetIdxs),
  2. keep x, y only and scale them to pixels (LandmarksToTensorCalculator),
  3. run face_blendshapes.tflite on a [1, 146, 2] float32 tensor,
  4. read 52 scores, labelled with kBlendshapeNames.
"""

import zipfile

import numpy as np
import pandas as pd                      
from ai_edge_litert.interpreter import Interpreter

# kLandmarksSubsetIdxs from face_blendshapes_graph.cc (order matters).
LANDMARKS_SUBSET_IDXS = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39,
    40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
    81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133,
    136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
    161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
    249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
    296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
    336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
    466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
]

# kBlendshapeNames from face_blendshapes_graph.cc.
BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]


def landmarks_to_array(face_landmarks):
    """Converts a list of NormalizedLandmark (one face) to a (478, 3) array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks],
                    dtype=np.float32)


class BlendshapePredictor:
    """Loads face_blendshapes.tflite from a face_landmarker.task bundle."""

    def __init__(self, task_path):
        with zipfile.ZipFile(task_path) as bundle:
            model_bytes = bundle.read("face_blendshapes.tflite")
        self._interpreter = Interpreter(model_content=model_bytes)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]

    def predict(self, landmarks, img_w, img_h):
        """Returns 52 blendshape scores.

        landmarks: (478, >=2) array of normalized x, y as MediaPipe returns them.
        img_w, img_h: size of the image the landmarks were computed on.
        """
        landmarks = np.asarray(landmarks, dtype=np.float32)
        xy = landmarks[LANDMARKS_SUBSET_IDXS, :2] * np.array(
            [img_w, img_h], dtype=np.float32)
        self._interpreter.set_tensor(self._input["index"], xy[None])
        self._interpreter.invoke()
        return self._interpreter.get_tensor(self._output["index"]).reshape(-1).copy()

    def create_bs_df(self, landmarks_per_frame, img_w, img_h):
        """Returns a DataFrame with a 'frame' column and one column per blendshape.

        landmarks_per_frame: sequence of per-frame landmark lists (NormalizedLandmark),
            e.g. mp_results_list or sam2_mp_results.
        """
        rows = []
        for frame_idx, face_landmarks in enumerate(landmarks_per_frame):
            scores = self.predict(landmarks_to_array(face_landmarks), img_w, img_h)
            rows.append([frame_idx, *scores])
        return pd.DataFrame(rows, columns=["frame"] + BLENDSHAPE_NAMES)
