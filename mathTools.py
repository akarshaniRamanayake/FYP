from scipy.spatial.distance import directed_hausdorff

def directed_hausdorff_distance(set_A, set_B):
    distance= directed_hausdorff(set_A, set_B)[0]
    return distance

# Function to determine if a new scan is accepted or rejected
def is_scan_accepted(new_scan, existing_dataset):
    similarity_threshold = 0.5  # Adjust as needed
    distance = directed_hausdorff_distance(new_scan, existing_dataset)
    #print("distance :  ",distance )
    return distance > similarity_threshold
