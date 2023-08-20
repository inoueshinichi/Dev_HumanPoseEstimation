"""姿勢推定結果表示用プログラム
"""
import os
import sys

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..'])
print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

########### Standard ###########

########### 3rd-parth ###########
import numpy as np
import cv2

########### Own ###########
from type_hint import *

from posenet.runtime_predict import (
    PoseNet,
    Point,
    KeyPoint,
    BBox,
    Person,
)

# map edges to a RGB color
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (147, 20, 255),
    (0, 2): (255, 255, 0),
    (1, 3): (147, 20, 255),
    (2, 4): (255, 255, 0),
    (0, 5): (147, 20, 255),
    (0, 6): (255, 255, 0),
    (5, 7): (147, 20, 255),
    (7, 9): (147, 20, 255),
    (6, 8): (255, 255, 0),
    (8, 10): (255, 255, 0),
    (5, 6): (0, 255, 255),
    (5, 11): (147, 20, 255),
    (6, 12): (255, 255, 0),
    (11, 12): (0, 255, 255),
    (11, 13): (147, 20, 255),
    (13, 15): (147, 20, 255),
    (12, 14): (255, 255, 0),
    (14, 16): (255, 255, 0)
}

COLOR_LIST = [
    (47, 79, 79),
    (139, 69, 19),
    (0, 128, 0),
    (0, 0, 139),
    (255, 0, 0),
    (255, 215, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (30, 144, 255),
    (255, 228, 181),
    (255, 105, 180),
]

def visualize(
    image: np.ndarray,
    list_persons: List[Person],
    keypoint_color: Tuple[int, ...] = None,
    keypoint_threshold: float = 0.05,
    instance_threshold: float = 0.1,
    ) -> np.ndarray:
    """Draws landmarks and edges on the input image and return it.

    Args:
        image: The input RGB image.
        list_persons: The list of all "Person" entities to be visualize.
        keypoint_color: the colors in which the landmarks should be plotted.
        keypoint_threshold: minimum confidence score for a keypoint to be drawn.
        instance_threshold: minimum confidence score for a person to be drawn.

    Returns:
        Image with keypoints and edges.
    """
    for person in list_persons:

        if person.score < instance_threshold:
            continue

        keypoints = person.keypoints
        bounding_box = person.bounding_box

        # Assign a color to visualize keypoints.
        if keypoint_color is None:
            if person.id is None:
                # If there's no person id, which means no tracker is enabled, use
                # a default color.
                person_color = (0, 255, 0)
            else:
                # If there's a person id, use different color for each person.
                person_color = COLOR_LIST[person.id % len(COLOR_LIST)]
        else:
            person_color = keypoint_color

        # Draw all the landmarks
        for i in range(len(keypoints)):
            if keypoints[i].score >= keypoint_threshold:
                cv2.circle(image, keypoints[i].coordinate, 2, person_color, 4)

        # Draw all the edges
        for edge_pair, edge_color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (keypoints[edge_pair[0]].score > keypoint_threshold and
                keypoints[edge_pair[1]].score > keypoint_threshold):
                cv2.line(image, keypoints[edge_pair[0]].coordinate,
                        keypoints[edge_pair[1]].coordinate, edge_color, 2)

            # Draw bounding_box with multipose
            if bounding_box is not None:
                start_point = bounding_box.ps
                end_point = bounding_box.pe
                cv2.rectangle(image, start_point, end_point, person_color, 2)
                # Draw id text when tracker is enabled for MoveNet MultiPose model.
                # (id = None when using single pose model or when tracker is None)
                if person.id:
                    id_text = 'id = ' + str(person.id)
                    cv2.putText(image, id_text, start_point, cv2.FONT_HERSHEY_PLAIN, 1,
                                (0, 0, 255), 1)

    return image