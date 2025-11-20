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

def hip_angle(left_shoulder, left_hip, left_knee):
    """Angle at the hip between shoulder and knee."""
    return angle(left_shoulder, left_hip, left_knee)

def knee_angle(left_hip, left_knee, left_ankle):
    """Angle at the knee between hip and ankle."""
    return angle(left_hip, left_knee, left_ankle)

def elbow_angle(left_shoulder, left_elbow, left_wrist):
    """Angle at the elbow between shoulder and wrist."""
    return angle(left_shoulder, left_elbow, left_wrist)

def torso_angle(left_shoulder, left_hip):
    """
    Computes the angle of the torso relative to vertical.
    Returns the absolute angle in degrees (0-90).
    """
    # Calculate horizontal and vertical distance between shoulder and hip
    dx = left_shoulder[0] - left_hip[0]  # horizontal displacement (x-axis)
    dy = left_shoulder[1] - left_hip[1]  # vertical displacement (y-axis)
    
    # arctan2(dx, dy) gives angle from vertical axis (not horizontal)
    # Note: args are swapped (dx, dy) instead of typical (dy, dx)
    # This measures lean/tilt: 0° = upright, 90° = horizontal
    angle_from_vertical = np.degrees(np.arctan2(dx, dy))
    
    # Return absolute value so angle is always positive (0-90°)
    return abs(angle_from_vertical)

def head_center(nose, left_ear, right_ear):
    """
    Computes the center of the head using ear positions.
    Ears are more stable landmarks than nose for helmet orientation.
    Returns the midpoint between left and right ears.
    """
    # Average x-coordinate of both ears (horizontal center)
    ex = (left_ear[0] + right_ear[0]) / 2
    # Average y-coordinate of both ears (vertical center)
    ey = (left_ear[1] + right_ear[1]) / 2
    return (ex, ey)

def gaze_vector(nose, head_center):
    """
    Computes the gaze direction vector from head center to nose.
    Represents where the rider is looking relative to head orientation.
    Returns a numpy array [dx, dy] pointing from head center toward nose.
    """
    nx, ny = nose
    hx, hy = head_center
    # Vector from head center to nose tip
    # Positive dx = looking right, Positive dy = looking down
    return np.array([nx - hx, ny - hy])

def gaze_angle(nose, left_ear, right_ear):
    """
    Computes the gaze angle relative to the horizontal axis.
    0° = looking straight right, 90° = looking down, -90° = looking up.
    """
    # Find the head center (midpoint between ears)
    hc = head_center(nose, left_ear, right_ear)
    # Get the gaze direction vector (nose relative to head)
    gx, gy = gaze_vector(nose, hc)
    
    # arctan2(gy, gx) gives angle from horizontal axis
    # Standard atan2 order: (y, x) measures from positive x-axis (right)
    # Result: -180° to +180°, where 0° = horizontal right
    angle = np.degrees(np.arctan2(gy, gx))
    return angle



