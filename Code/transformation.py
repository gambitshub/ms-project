"""
transformation.py

Author: James Daniels

This file contains the random transformation generator
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def random_rotation_matrix(seed: int, 
                           range_r: float) -> np.ndarray:
    """
    Generate a random 3D rotation matrix.

    Parameters:
    - seed (int): The seed for the random number generator. Ensures reproducibility.
    - range_r (float): The maximum range of the rotation in degrees.

    Returns:
    - np.ndarray: A 3x3 rotation matrix.
    """
    # Set the seed for the random number generator to ensure reproducibility
    np.random.seed(seed)

    # Convert the range to radians
    range_r = np.radians(range_r)

    # Generate a random angle between 0 and range_r. This is the magnitude of the rotation.
    angle = np.random.uniform(0, range_r)
    # print(angle)

    # Generate a random 3D vector with components between -1 and 1. This vector represents the axis of rotation in 3D space.
    axis = np.random.uniform(-1, 1, 3)

    # Normalize the vector to make it a unit vector
    axis /= np.linalg.norm(axis)

    # Compute the four parameters of the quaternion representation of the rotation. Quaternions are a way to represent rotations in 3D space.
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)

    # Convert the quaternion representation to a rotation matrix and return it
    rotation_matrix = np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]
    ])

    # print(f"Generated rotation matrix with seed {seed} and range {range_r}:\n{rotation_matrix}")
    return rotation_matrix


def random_translation_vector(seed: int, 
                              range_t: float) -> np.ndarray:
    """
    Generate a random 3D translation vector.

    Parameters:
    - seed (int): The seed for the random number generator. This ensures reproducibility.
    - range_t (float): The maximum range of the translation. The translation vector components will be in the interval [-range_t, range_t].

    Returns:
    - np.ndarray: A 3D translation vector.
    """
    # Set the seed for the random number generator to ensure reproducibility
    np.random.seed(seed)

    # Generate a random 3D translation vector with components between -range_t and range_t
    translation_vector = np.random.uniform(-range_t, range_t, 3)

    # print(f"Generated translation vector with seed {seed} and range {range_t}:\n{translation_vector}")
    return translation_vector


def random_transformation_matrix(idx: int, 
                                 range_t: float, 
                                 range_r: float) -> np.ndarray:
    """
    Generate a random 4x4 transformation matrix.

    Parameters:
    - idx (int): The index to use as a seed for the random number generator. This ensures that the same transformation matrix is generated for the same index.
    - range_t (float): The maximum range of the translation. The translation vector components will be in the interval [-range_t, range_t].
    - range_r (float): The maximum range of the rotation in degrees. The rotation will be a random angle in the interval [0, range_r].

    Returns:
    - np.ndarray: A 4x4 transformation matrix.
    """
    # Generate a random 3D rotation matrix using the index as a seed and range_r as the maximum rotation range
    rotation_matrix = random_rotation_matrix(idx, range_r)

    # Generate a random 3D translation vector using the index as a seed and range_t as the maximum translation range
    translation_vector = random_translation_vector(idx, range_t)

    # Create an identity matrix
    transformation_matrix = np.identity(4)

    # Replace the upper left 3x3 part of the identity matrix with the rotation matrix
    transformation_matrix[:3, :3] = rotation_matrix

    # Replace the last column of the identity matrix with the translation vector
    transformation_matrix[:3, 3] = translation_vector

    # r, theta, translation_vector = extract_parameters_from_transformation(transformation_matrix)
    # print('angle extracted')
    # print(theta)
    # print('axis extracted')
    # print(r)
    # print('translation extracted')
    # print(translation_vector)

    # rotation_gt, t1 = extract_rotation_translation(transformation_matrix)
    # axis, angle_gt = extract_rotation_parameters(rotation_gt)
    # print('\n angle gt = ')
    # print(angle_gt)
    # print(t1)
    # print(axis)

    return transformation_matrix


def extract_rotation_translation(transformation_matrix: np.ndarray):
    """
    Extract the rotation matrix and translation vector from a 4x4 transformation matrix.

    Parameters:
    - transformation_matrix (np.ndarray): The 4x4 transformation matrix.

    Returns:
    - np.ndarray: The 3x3 rotation matrix.
    - np.ndarray: The 3D translation vector.
    """
    # The rotation matrix is the top left 3x3 part of the transformation matrix
    rotation_matrix = transformation_matrix[:3, :3].copy()

    # The translation vector is the rightmost column of the transformation matrix
    translation_vector = transformation_matrix[:3, 3].copy()

    return rotation_matrix, translation_vector


def extract_rotation_parameters(rotation_matrix: np.ndarray):
    """
    Extract the axis of rotation and the angle of rotation from a 3x3 rotation matrix.

    Parameters:
    - rotation_matrix (np.ndarray): The 3x3 rotation matrix.

    Returns:
    - np.ndarray: The axis of rotation as a unit vector.
    - float: The angle of rotation in radians.
    """
    # print(f"this is the rotation matrix:\n{rotation_matrix}")
    # Compute the angle of rotation
    angle_of_rotation = np.arccos((np.trace(rotation_matrix) - 1) / 2)

    # Compute the axis of rotation
    axis_of_rotation = np.array([
        rotation_matrix[2, 1] - rotation_matrix[1, 2],
        rotation_matrix[0, 2] - rotation_matrix[2, 0],
        rotation_matrix[1, 0] - rotation_matrix[0, 1]
    ])
    axis_of_rotation = axis_of_rotation / (2.0 * np.sin(angle_of_rotation))

    return axis_of_rotation, angle_of_rotation


def extract_parameters_from_transformation(transformation_matrix):
    """
    Extracts the rotation and translation parameters from the given transformation matrix.
    The rotation is returned as an axis-angle representation.

    Parameters:
    - transformation_matrix (np.ndarray): The 4x4 transformation matrix.

    Returns:
    - np.ndarray: The axis of rotation.
    - float: The angle of rotation.
    - np.ndarray: The translation vector.
    """
    # Extract the rotation matrix and create a copy of it
    rotation_matrix = transformation_matrix[:3, :3].copy()

    # Extract the translation vector
    translation_vector = transformation_matrix[:3, 3].copy()

    # Convert the rotation matrix to a rotation vector
    rotation = R.from_matrix(rotation_matrix)
    rotvec = rotation.as_rotvec()

    # Compute the axis and angle
    axis = rotvec / np.linalg.norm(rotvec)
    angle = np.linalg.norm(rotvec)

    # Check if the rotation angle is negative
    if angle < 0:
        # If the rotation angle is negative, invert the axis and make the angle positive
        axis = -axis
        angle = -angle

    # Rodrigues' rotation formula for better numerical stability
    theta = np.arccos((np.trace(rotation_matrix) - 1) / 2)
    r = 1 / (2 * np.sin(theta)) * np.array([rotation_matrix[2, 1] - rotation_matrix[1, 2], rotation_matrix[0, 2] - rotation_matrix[2, 0], rotation_matrix[1, 0] - rotation_matrix[0, 1]])

    # Normalizing the rotation axis vector
    r = r / np.linalg.norm(r)

    return r, theta, translation_vector