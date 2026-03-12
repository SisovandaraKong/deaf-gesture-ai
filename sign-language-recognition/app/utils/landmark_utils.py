"""
app/utils/landmark_utils.py

Landmark normalization helper — identical to the normalization logic
in the original realtime_recognition.py and extract_landmarks.py.

Each landmark's (x, y) is shifted so that the bounding-box origin
becomes (0, 0), making the representation scale/position-invariant.
"""


def normalize_landmarks(hand_landmarks) -> list:
    """
    Normalize a list of MediaPipe NormalizedLandmark objects relative
    to their bounding-box top-left corner.

    Matches original logic:
        landmarks.append(lm.x - min(x_coords))
        landmarks.append(lm.y - min(y_coords))

    Args:
        hand_landmarks: list of MediaPipe NormalizedLandmark (21 points)

    Returns:
        Flat list of 42 floats [x0, y0, x1, y1, ..., x20, y20]
    """
    x_coords = [lm.x for lm in hand_landmarks]
    y_coords = [lm.y for lm in hand_landmarks]
    x_min = min(x_coords)
    y_min = min(y_coords)

    landmarks = []
    for lm in hand_landmarks:
        landmarks.append(lm.x - x_min)
        landmarks.append(lm.y - y_min)

    return landmarks
