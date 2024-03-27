import datetime
import numpy as np
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from scipy.spatial import cKDTree
import math
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from mathTools import is_scan_accepted

class Lidar():
    def __init__(self, GUI):
        self.GUI = GUI
        self.nuScenes_data_path = '/Users/ak/PycharmProjects/SDF/v1.0-mini/'
        self.nusc = NuScenes('v1.0-mini', dataroot= self.nuScenes_data_path, verbose=True)

        # Get initial point cloud
        self.sample = self.nusc.sample[0]
        self.lidar_sensor_token = self.sample['data']['LIDAR_TOP']
        self.lidar_data = self.nusc.get('sample_data', self.lidar_sensor_token)
        self.lidar_filepath = self.nuScenes_data_path + '/' + self.lidar_data['filename']
        self.lidar_points = np.fromfile(self.lidar_filepath, dtype=np.float32).reshape(-1, 5)
        self.existing_points_coordinates = self.lidar_points[:, :3].tolist()  # x, y, z coordinates
        self.max_dataset_size = len(self.existing_points_coordinates)

    def load(self):
        print(len(self.nusc.sample))
        for i in range(len(self.nusc.sample)):
            sample_index = i
            sample = self.nusc.sample[sample_index]

            # Get lidar sensor information
            lidar_sensor_token = sample['data']['LIDAR_TOP']
            lidar_data = self.nusc.get('sample_data', lidar_sensor_token)

            # Load lidar point cloud data
            lidar_filepath = self.nuScenes_data_path + '/' + lidar_data['filename']
            lidar_points = np.fromfile(lidar_filepath, dtype=np.float32).reshape(-1, 5)

            # Extract point cloud details
            new_points_coordinates = lidar_points[:, :3].tolist()  # x, y, z coordinates

            if is_scan_accepted(new_points_coordinates, self.existing_points_coordinates):
                self.existing_points_coordinates = np.vstack((self.existing_points_coordinates, new_points_coordinates))
                if self.existing_points_coordinates.shape[0] > self.max_dataset_size:
                    self.existing_points_coordinates = self.existing_points_coordinates[-self.max_dataset_size:]

                if (self.GUI.ifBreak):
                    break

                self.GUI.pointCoordinates = self.existing_points_coordinates
                self.GUI.updateScene.emit("New scene loaded")
