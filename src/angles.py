import numpy as np

def angle(a, b, c):
    """
    Computes the angle ABC (in degrees)
    where:
    - a, b, c are (x, y)
    - angle at b, which is the vertex
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def hip_angle(left_hip, left_shoulder, left_knee):
    return angle(left_shoulder, left_hip, left_knee)

def knee_angle(left_hip, left_knee, left_ankle):
    return angle(left_hip, left_knee, left_ankle)

def elbow_angle(left_shoulder, left_elbow, left_wrist):
    return angle(left_shoulder, left_elbow, left_wrist)

def torso_angle(left_shoulder, left_hip):
    dx = left_shoulder[0] - left_hip[0]
    dy = left_shoulder[1] - left_hip[1]
    angle_from_vertical = np.degrees(np.arctan2(dx, dy))
    return abs(angle_from_vertical)


