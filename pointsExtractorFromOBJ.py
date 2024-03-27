def extract_points_from_obj():
    file_path = '/Users/ak/PycharmProjects/SDF/RESOURCES/uploads_files_2787791_Mercedes+Benz+GLS+580.obj'
    points = []

    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):  # 'v ' denotes vertex data
                parts = line.split()[1:]  # Remove the 'v' identifier
                point = tuple(map(float, parts))  # Convert strings to float and create a tuple
                points.append(point)

    return points

def bounding_box_corners(points):
    if not points:
        return None

    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

    for point in points:
        x, y, z = point
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)

    return [
        (min_x, min_y, min_z),
        (max_x, min_y, min_z),
        (max_x, max_y, min_z),
        (min_x, max_y, min_z),
        (min_x, min_y, max_z),
        (max_x, min_y, max_z),
        (max_x, max_y, max_z),
        (min_x, max_y, max_z)
    ]
