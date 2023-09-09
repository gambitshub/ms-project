"""
transformation.py

Author: James Daniels

This file contains the random transformation generator
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

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
    rotation_matrix = transformation_matrix[:3, :3]

    # The translation vector is the rightmost column of the transformation matrix
    translation_vector = transformation_matrix[:3, 3]

    return rotation_matrix, translation_vector


# def extract_rotation_parameters(rotation_matrix: np.ndarray):
#     """
#     Extract the axis of rotation and the angle of rotation from a 3x3 rotation matrix.

#     Parameters:
#     - rotation_matrix (np.ndarray): The 3x3 rotation matrix.

#     Returns:
#     - np.ndarray: The axis of rotation as a unit vector.
#     - float: The angle of rotation in radians.
#     """
#     # Perform eigenvalue decomposition
#     eigenvalues, eigenvectors = np.linalg.eig(rotation_matrix)

#     # The axis of rotation corresponds to the eigenvector associated with the eigenvalue 1
#     axis_of_rotation = eigenvectors[:, np.isclose(eigenvalues, 1)].flatten()

#     # The angle of rotation can be derived from the other two eigenvalues
#     angle_of_rotation = np.arccos((np.trace(rotation_matrix) - 1) / 2)

#     print('axis of rotation extracted')
#     print(axis_of_rotation)

#     print('angle of rotation extracted')
#     print(angle_of_rotation)

#     return axis_of_rotation, angle_of_rotation


# def extract_rotation_parameters(rotation_matrix):
#     """
#     Extract the axis of rotation and the angle of rotation from a 3x3 rotation matrix.

#     Parameters:
#     - rotation_matrix (np.ndarray): The 3x3 rotation matrix.

#     Returns:
#     - np.ndarray: The axis of rotation as a unit vector.
#     - float: The angle of rotation in radians.
#     """
#     rotation = R.from_matrix(rotation_matrix)
#     rotvec = rotation.as_rotvec()
#     axis = rotvec / np.linalg.norm(rotvec)
#     angle = np.linalg.norm(rotvec)
#     return axis, angle


def extract_rotation_parameters(rotation_matrix: np.ndarray):
    """
    Extract the axis of rotation and the angle of rotation from a 3x3 rotation matrix.

    Parameters:
    - rotation_matrix (np.ndarray): The 3x3 rotation matrix.

    Returns:
    - np.ndarray: The axis of rotation as a unit vector.
    - float: The angle of rotation in radians.
    """
    print(f"this is the rotation matrix:\n{rotation_matrix}")
    # Compute the angle of rotation
    angle_of_rotation = np.arccos((np.trace(rotation_matrix) - 1) / 2)

    # Compute the axis of rotation
    axis_of_rotation = np.array([
        rotation_matrix[2, 1] - rotation_matrix[1, 2],
        rotation_matrix[0, 2] - rotation_matrix[2, 0],
        rotation_matrix[1, 0] - rotation_matrix[0, 1]
    ])
    axis_of_rotation /= np.linalg.norm(axis_of_rotation)

    # axis_of_rotation = axis_of_rotation / (2.0 * np.sin(angle_of_rotation))

    print('axis of rotation extracted')
    print(axis_of_rotation)

    print('angle of rotation extracted')
    print(angle_of_rotation)

    return axis_of_rotation, angle_of_rotation


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
    rotation_matrix, axis_og, angle_og = random_rotation_matrix(idx, range_r)

    # Generate a random 3D translation vector using the index as a seed and range_t as the maximum translation range
    translation_vector = random_translation_vector(idx, range_t)

    # print('this is the translation vector')
    # print(translation_vector)

    # Create an identity matrix
    transformation_matrix = np.identity(4)

    # Replace the upper left 3x3 part of the identity matrix with the rotation matrix
    transformation_matrix[:3, :3] = rotation_matrix

    # Replace the last column of the identity matrix with the translation vector
    transformation_matrix[:3, 3] = translation_vector

    # axis, angle, translation = extract_parameters_from_transformation(transformation_matrix)
    # print('random axis =')
    # print(axis)
    # print('random angle =')
    # print(angle)
    # print('random trans =')
    # print(translation)
    # are_rotations_same = compare_rotations(axis, angle, axis_og, angle_og)
    # print(are_rotations_same)

    return transformation_matrix


from scipy.spatial.transform import Rotation as R

# def random_rotation_matrix(seed: int, 
#                            range_r: float) -> np.ndarray:
#     """
#     Generate a random 3D rotation matrix.

#     Parameters:
#     - seed (int): The seed for the random number generator. Ensures reproducibility.
#     - range_r (float): The maximum range of the rotation in degrees.

#     Returns:
#     - np.ndarray: A 3x3 rotation matrix.
#     """
#     # Set the seed for the random number generator to ensure reproducibility
#     np.random.seed(seed)

#     range_r = np.radians(range_r)

#     # Generate a random angle between 0 and range_r. This is the magnitude of the rotation.
#     angle = np.random.uniform(0, range_r)

#     # print('random angle = ')
#     # print(angle)

#     # Generate a random 3D vector with components between -1 and 1. This vector represents the axis of rotation in 3D space.
#     axis = np.random.uniform(-1, 1, 3)

#     # print('axis = ')
#     # print(axis)

#     # Normalize the vector to make it a unit vector
#     axis /= np.linalg.norm(axis)

#     # Construct the rotation object
#     rotation = R.from_rotvec(axis * angle)

#     # Convert the rotation to matrix form
#     rotation_matrix = rotation.as_matrix()

#     # print(f"Generated rotation matrix with seed {seed} and range {range_r}:\n{rotation_matrix}")
#     return rotation_matrix



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

    range_r = np.radians(range_r)

    # Generate a random angle between 0 and range_r. This is the magnitude of the rotation.
    angle = np.random.uniform(0, range_r)

    print('random angle = ')
    print(angle)

    # Generate a random 3D vector with components between -1 and 1. This vector represents the axis of rotation in 3D space.
    axis = np.random.uniform(-1, 1, 3)

    print('axis = ')
    print(axis)

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

    print(f"Generated rotation matrix with seed {seed} and range {range_r}:\n{rotation_matrix}")
    return rotation_matrix


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


def compare_rotations(axis1: np.ndarray, angle1: float,
                      axis2: np.ndarray, angle2: float) -> bool:
    """
    Compare two rotations represented by rotation axes and angles. 

    Parameters:
    - axis1 (np.ndarray): The rotation axis of the first rotation.
    - angle1 (float): The rotation angle of the first rotation.
    - axis2 (np.ndarray): The rotation axis of the second rotation.
    - angle2 (float): The rotation angle of the second rotation.

    Returns:
    - bool: True if the rotations are similar, False otherwise.
    """
    # Define a list of test vectors
    test_vectors = [np.array([1, 0, 0]), 
                    np.array([0, 1, 0]), 
                    np.array([0, 0, 1]), 
                    np.array([1, 1, 1]), 
                    np.array([-1, -1, -1]), 
                    np.array([1, -1, 0])]

    # Convert the axes and angles to rotation objects
    rot1 = R.from_rotvec(axis1 * angle1)
    rot2 = R.from_rotvec(axis2 * angle2)

    # Loop over the test vectors
    for test_vector in test_vectors:
        # Apply the rotations to the test vector
        result1 = rot1.apply(test_vector)
        result2 = rot2.apply(test_vector)

        # If the results are not close, return False
        if not np.allclose(result1, result2, atol=1e-6):
            return False

    # If all results are close, return True
    return True