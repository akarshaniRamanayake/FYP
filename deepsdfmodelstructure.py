import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F
from scipy.spatial import cKDTree

# Define the model architecture
class SDFConfidenceNetwork(nn.Module):
    def __init__(self):
        super(SDFConfidenceNetwork, self).__init__()

        # Define the Fourier feature layer
        self.fourier_layer = nn.Linear(3, 128)

        # SDF model
        self.sdf_model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Confidence model
        self.confidence_model = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Fourier feature encoding
        x = torch.relu(self.fourier_layer(x))

        # SDF prediction
        sdf_output = self.sdf_model(x)

        # Confidence prediction
        confidence_output = torch.tanh(self.confidence_model(x))

        return sdf_output, confidence_output

##########################################data pre processing ###############################################
def augment_data(lidar_data, surface_data, max_distance, b):
    # Assuming lidar_data is a tensor with shape (batch_size, num_points, 3)
    # and surface_data is a tensor with shape (batch_size, num_surface_points, 3)

    # Uniformly sample points along each LiDAR ray
    sampled_points = sample_points_along_rays(lidar_data)

    # Calculate approximate depth and confidence labels
    sdf_labels, confidence_labels = calculate_labels(sampled_points, surface_data, max_distance, b)

    # Concatenate original and augmented data
    augmented_data = torch.cat([lidar_data, sampled_points], dim=1)

    return augmented_data, sdf_labels, confidence_labels

def sample_points_along_rays(lidar_data):
    # Your sampling logic here
    # This could involve uniform sampling along rays or any other sampling strategy
    # The resulting tensor should have shape (batch_size, num_points, 3)

    # For demonstration purposes, let's assume uniform sampling along rays
    num_points = lidar_data.size(1)
    sampled_points = lidar_data + torch.rand_like(lidar_data) * 0.1  # Adjust the factor as needed

    return sampled_points

def calculate_labels(sampled_points, surface_data, max_distance, b):
    # Calculate SDF and confidence labels based on the described procedure

    # Build a KD-Tree with surface points
    surface_tree = cKDTree(surface_data.cpu().numpy())

    # Calculate SDF values for free space points
    sdf_labels = calculate_sdf_labels(sampled_points, surface_tree)

    # Calculate confidence labels based on distance and hyperparameter b
    confidence_labels = calculate_confidence_labels(sampled_points, surface_tree, max_distance, b)

    return sdf_labels, confidence_labels

def calculate_sdf_labels(sampled_points, surface_tree):
    # Calculate SDF values for free space points

    # Query KD-Tree to find the distance to the nearest surface point
    distances, _ = surface_tree.query(sampled_points.cpu().numpy(), k=1)

    # Convert distances to PyTorch tensor
    sdf_labels = torch.from_numpy(distances).to(sampled_points.device).float()

    return sdf_labels

def calculate_confidence_labels(sampled_points, surface_tree, max_distance, b):
    # Calculate confidence labels based on distance and hyperparameter b

    # Query KD-Tree to find the distance to the nearest surface point
    distances, _ = surface_tree.query(sampled_points.cpu().numpy(), k=1)

    # Calculate normalized distance between 0 and 1
    normalized_distances = 1.0 - distances / max_distance
    normalized_distances = torch.clamp(normalized_distances, 0.0, 1.0)

    # Exponentially decreasing confidence labels
    confidence_labels = torch.exp(-b * normalized_distances)

    # Assign confidence value 1 to points with SDF >= 0
    sdf_values = calculate_sdf_labels(sampled_points, surface_tree)
    confidence_labels[sdf_values >= 0] = 1.0

    return confidence_labels

# # Example usage
# lidar_data = torch.randn((32, 512, 3))  # Assuming a batch size of 32 and 512 points per sample
# surface_data = torch.randn((32, 1024, 3))  # Assuming a batch size of 32 and 1024 surface points
# max_distance = 10.0  # Adjust as needed
# b = 5.0  # Adjust as needed
#
# augmented_data, sdf_labels, confidence_labels = augment_data(lidar_data, surface_data, max_distance, b)

########################################## Data ingestion functions##########################################
def ingest_and_augment_data(new_scan, current_surface_data, current_synthetic_data, max_dataset_size):
    # Calculate directed Hausdorff distance to determine similarity
    similarity_score = calculate_similarity(new_scan, current_surface_data)

    # Define a threshold for accepting or rejecting the new scan
    similarity_threshold = 0.5  # Adjust as needed

    if similarity_score > similarity_threshold:
        # Separate the new scan into obstacle and floor points
        obstacle_points, floor_points = separate_obstacle_floor(new_scan)

        # Discard floor points
        # You may need to adjust this based on your specific data structure
        new_scan = obstacle_points

        # Generate synthetic data by shooting rays from the current robot origin
        synthetic_data = generate_synthetic_data(new_scan)

        # Calculate signed distance labels
        signed_distance_labels = calculate_signed_distance_labels(new_scan, current_surface_data)

        # Generate confidence labels
        confidence_labels = generate_confidence_labels(new_scan, current_surface_data)

        # Concatenate new and old surface data
        updated_surface_data = torch.cat([current_surface_data, new_scan], dim=1)

        # Concatenate new and old synthetic data
        updated_synthetic_data = torch.cat([current_synthetic_data, synthetic_data], dim=1)

        # Sub-sample the combined dataset to the desired size
        updated_surface_data, updated_synthetic_data = sub_sample_dataset(updated_surface_data, updated_synthetic_data, max_dataset_size)

        return updated_surface_data, updated_synthetic_data, signed_distance_labels, confidence_labels

    else:
        # If the new scan is not accepted, return the existing data
        return current_surface_data, current_synthetic_data, None, None

def calculate_similarity(new_scan, current_surface_data):
    # Your similarity calculation logic here
    # This could be the directed Hausdorff distance or any other similarity metric
    similarity_score = 0.8  # Replace with your actual similarity calculation

    return similarity_score

def separate_obstacle_floor(scan):
    # Your separation logic here
    # Assuming scan is a tensor with shape (batch_size, num_points, 3)
    obstacle_points = scan[scan[:, :, 2] > threshold_obstacle_height]
    floor_points = scan[scan[:, :, 2] <= threshold_obstacle_height]

    return obstacle_points, floor_points

def generate_synthetic_data(scan):
    # Your synthetic data generation logic here
    # This could involve shooting rays from the robot origin and sampling along them
    synthetic_data = scan  # Replace with your actual logic

    return synthetic_data

def calculate_signed_distance_labels(scan, current_surface_data):
    # Your signed distance label calculation logic here
    # This could involve finding the distance to the nearest surface point for each sample
    signed_distance_labels = torch.zeros_like(scan[:, :, 0])  # Replace with your actual logic

    return signed_distance_labels

def generate_confidence_labels(scan, current_surface_data):
    # Your confidence label generation logic here
    # This could involve the weighting process described in your description
    confidence_labels = torch.ones_like(scan[:, :, 0])  # Replace with your actual logic

    return confidence_labels

def sub_sample_dataset(surface_data, synthetic_data, max_dataset_size):
    # Your sub-sampling logic here
    # This could involve uniformly choosing points at random until the dataset is of the desired size
    # Be mindful of keeping corresponding surface and synthetic data points aligned
    indices = torch.randperm(surface_data.size(1))[:max_dataset_size]

    return surface_data[:, indices, :], synthetic_data[:, indices, :]

# Example usage
# new_scan = torch.randn((32, 512, 3))  # Assuming a batch size of 32 and 512 points per new scan
# current_surface_data = torch.randn((32, 1024, 3))  # Assuming a batch size of 32 and 1024 points in the current surface data
# current_synthetic_data = torch.randn((32, 1024, 3))  # Assuming a batch size of 32 and 1024 points in the current synthetic data
# max_dataset_size = 1000
#
# updated_surface_data, updated_synthetic_data, signed_distance_labels, confidence_labels = \
#     ingest_and_augment_data(new_scan, current_surface_data, current_synthetic_data, max_dataset_size)


############################################ Local SDF training function ##############################################
def train_local_sdf(model, labeled_data, convergence_threshold=0.001, max_epochs=100):
    device = next(model.parameters()).device

    # Unpack labeled_data
    spatial_locations, sdf_labels, confidence_labels = labeled_data

    # Convert data to PyTorch tensors
    spatial_locations = torch.tensor(spatial_locations, dtype=torch.float32, device=device)
    sdf_labels = torch.tensor(sdf_labels, dtype=torch.float32, device=device)
    confidence_labels = torch.tensor(confidence_labels, dtype=torch.float32, device=device)

    # Define the loss function
    delta = 0.1  # Huber loss parameter
    loss_fn_sdf = nn.SmoothL1Loss(delta=delta)
    loss_fn_confidence = nn.SmoothL1Loss()

    # Eikonal loss
    def eikonal_loss(gradient):
        return torch.mean((torch.norm(gradient, dim=-1) - 1) ** 2)

    # Regularization loss
    def regularization_loss(weights):
        return 1e-3 * torch.max(weights)

    # Set the model in training mode
    model.train()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(max_epochs):
        # Forward pass
        sdf_predictions, confidence_predictions = model(spatial_locations)

        # Compute losses
        loss_sdf = loss_fn_sdf(sdf_predictions, sdf_labels)
        loss_confidence = loss_fn_confidence(confidence_predictions, confidence_labels)
        gradient_sdf = torch.autograd.grad(loss_sdf, spatial_locations, create_graph=True)[0]
        loss_eikonal = eikonal_loss(gradient_sdf)
        loss_regularization = regularization_loss(model.parameters())

        # Total loss
        total_loss = 1e4 * (loss_sdf + loss_confidence) + loss_eikonal + loss_regularization

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Check for convergence
        if total_loss.item() < convergence_threshold:
            break

    # Save the trained model
    torch.save(model.state_dict(), "local_sdf_model.pth")

    return model

# Example usage:
# Assuming labeled_data is a tuple (spatial_locations, sdf_labels, confidence_labels)
# And model is an instance of your SDF model
# train_local_sdf(model, labeled_data)



######################################### Global map construction function ############################################
def construct_global_map(local_maps, query_points):
    # Assuming local_maps is a list of trained local SDF models
    # and query_points is a tensor of spatial locations where the global map is queried

    global_sdf_predictions = torch.zeros_like(query_points[:, 0])  # Initialize global SDF predictions
    global_confidence_predictions = torch.zeros_like(query_points[:, 0])  # Initialize global confidence predictions

    for local_map in local_maps:
        # Evaluate the local SDF model at query points
        local_sdf_predictions, local_confidence_predictions = local_map(query_points)

        # Update global predictions based on local map confidence
        update_mask = local_confidence_predictions > global_confidence_predictions
        global_sdf_predictions[update_mask] = local_sdf_predictions[update_mask]
        global_confidence_predictions[update_mask] = local_confidence_predictions[update_mask]

    return global_sdf_predictions, global_confidence_predictions

# Example usage:
# Assuming local_maps is a list of trained local SDF models
# And query_points is a tensor of spatial locations where the global map is queried
# global_sdf_predictions, global_confidence_predictions = construct_global_map(local_maps, query_points)

# Example usage
model = SDFConfidenceNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming you have a PyTorch DataLoader for your dataset
for epoch in range(num_epochs):
    for batch in dataloader:
        # Data pre-processing
        augmented_data = augment_data(batch)

        # Data ingestion
        current_data = ingest_data(new_scan, current_data)

        # Local SDF training
        sdf_output, confidence_output = model(augmented_data)
        loss = your_loss_function(sdf_output, confidence_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Global map construction
global_map = construct_global_map(local_maps)
