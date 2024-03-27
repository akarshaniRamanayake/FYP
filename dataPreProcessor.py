import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader

from scipy.spatial import cKDTree

# hyper parameters
confidence_param = 2
cKDLeafs_param = 50

def preProcess(pointCoordinates):
    distances = []
    confidences = []

    #floor_points = pointCoordinates[pointCoordinates[:, 2] < 1.562]
    obstacle_points = pointCoordinates[pointCoordinates[:, 2] >= 1.562].tolist()

    # Build a K-D Tree
    kdtree = cKDTree(obstacle_points, leafsize=cKDLeafs_param)

    completed_dataset = []
    positive_points = []
    negative_points = []
    for index in range(len(obstacle_points)):
        uniform_values_positive = np.random.rand(10)
        uniform_values_negative = sorted(np.random.rand(10))
        obstacle_point = obstacle_points[index]

        max_minus_dist = math.sqrt(pow(uniform_values_negative[-1] * obstacle_point[0], 2) + pow(
            uniform_values_negative[-1] * obstacle_point[1], 2) + pow(
            uniform_values_negative[-1] * obstacle_point[2], 2))

        # Add sign distance for distance plus points (points outside obstacles) and add confidence
        for i in range(len(uniform_values_positive)):
            new_point = [uniform_values_positive[i] * obstacle_point[0],
                         uniform_values_positive[i] * obstacle_point[1],
                         uniform_values_positive[i] * obstacle_point[2]]
            positive_points.append(new_point)
            distance, nearest_surface_point_index = kdtree.query(new_point, k=1)
            new_point.append(abs(distance))
            new_point.append(1)
            completed_dataset.append(new_point)
            distances.append(abs(distance))
            confidences.append(1)

        # Add sign distance for distance minus points (points inside obstacles)
        for i in range(len(uniform_values_negative)):
            negative_x = obstacle_point[0] + uniform_values_negative[i] * obstacle_point[0]
            negative_y = obstacle_point[1] + uniform_values_negative[i] * obstacle_point[1]
            negative_z = obstacle_point[2] + uniform_values_negative[i] * obstacle_point[2]
            new_point = [negative_x, negative_y, negative_z]
            negative_points.append(new_point)
            distance, nearest_surface_point_index = kdtree.query(new_point, k=1)
            new_point.append((-1) * abs(distance))

            # Negative points confidence calculation.
            dist_along_ray = math.sqrt(pow(uniform_values_negative[i] * obstacle_point[0], 2)
                                       + pow(uniform_values_negative[i] * obstacle_point[1], 2)
                                       + pow(uniform_values_negative[i] * obstacle_point[2], 2))
            confidence = ((pow(confidence_param, 1 - dist_along_ray / max_minus_dist) - 1) / (
                    confidence_param - 1)) + 1e-7
            new_point.append(confidence)
            distances.append((-1) * abs(distance))
            confidences.append(confidence)

            completed_dataset.append(new_point)

        pnt = []
        pnt.append(obstacle_point[0])
        pnt.append(obstacle_point[1])
        pnt.append(obstacle_point[2])
        pnt.append(0)
        pnt.append(1)
        distances.append(0)
        confidences.append(1)
        completed_dataset.append(pnt)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(distances, confidences, c=['black'], s=1)
    ax.set_title('Confidence vs Distance', fontsize=16)
    ax.set_xlabel('Distance', fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.savefig('confidence_vs_distance.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    # Convert to numpy array
    completed_dataset = np.array(completed_dataset)
    print("size", completed_dataset.size)
    return obstacle_points, completed_dataset

def loadData(completed_dataset):
    # Split features (independent variables) and labels (dependent variables)
    independent = completed_dataset[:, :3]  # Features
    dependent = completed_dataset[:, 3:]  # Labels

    # Convert to PyTorch tensors
    independent_tensor = torch.tensor(independent, dtype=torch.float32)
    dependent_tensor = torch.tensor(dependent, dtype=torch.float32)

    # Combine features and labels into a TensorDataset
    dataset = TensorDataset(independent_tensor, dependent_tensor)

    # Define the sizes of train and test datasets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Set random seed for reproducibility
    torch.manual_seed(48)

    # Split dataset into train and test
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader,test_loader
