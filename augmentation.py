import os
import cv2
import json
import datetime
import numpy as np
from shapely.geometry import Polygon, LineString

# ===========================================================================================================================
def convert_mask_to_polygon(mask):
    """Converts a binary mask image into a list of polygons.

    Args:
        mask (numpy.ndarray): The input binary mask image.

    Returns:
        list: A list of polygons, where each polygon is represented as a numpy array of vertices.
    """
    # Convert the mask to binary
    _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over the contours and approximate to polygons
    polygons = []
    for contour in contours:
        # Approximate the contour to a polygon with epsilon = 0.001 * perimeter
        epsilon = 0.005 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        polygons.append(polygon.reshape(polygon.shape[0], polygon.shape[2]))
    
    return polygons

# ===========================================================================================================================
def line_polygon_intersection(line_start, line_end, polygon_points):
    """Finds the intersection points between a line and a polygon.

    Args:
        line_start (tuple): The starting point of the line in the format (x, y).
        line_end (tuple): The ending point of the line in the format (x, y).
        polygon_points (list): A list of tuples representing the vertices of the polygon [(x1, y1), (x2, y2), ...].

    Returns:
        list: A list of tuples representing the intersection points [(x1, y1), (x2, y2), ...].
    """
    # Create a LineString object representing the line
    line = LineString([line_start, line_end])

    # Create a Polygon object representing the polygon
    polygon = Polygon(polygon_points)
    
    if line.intersects(polygon):
        intersection = line.intersection(polygon)
        if str(type(intersection)) == "<class 'shapely.geometry.multilinestring.MultiLineString'>":
            result = []
            for line in intersection.geoms:
                result += list(line.coords)
            return result
        return list(intersection.coords)
    return []

# ===========================================================================================================================
def lines_intersection(line_1_start, line_1_end, line_2_start, line_2_end):
    """Finds the intersection point of two lines.

    Args:
        line_1_start (tuple): The starting point of line 1 in the format (x, y).
        line_1_end (tuple): The ending point of line 1 in the format (x, y).
        line_2_start (tuple): The starting point of line 2 in the format (x, y).
        line_2_end (tuple): The ending point of line 2 in the format (x, y).

    Returns:
        tuple: The coordinates (x, y) of the intersection point if it exists, or None if the lines do not intersect.
    """
    line1 = LineString([line_1_start, line_1_end])
    line2 = LineString([line_2_start, line_2_end])
    
    intersection = line1.intersection(line2)
    
    if intersection.is_empty:
        return None
    else:
        return intersection.x, intersection.y
    
# ===========================================================================================================================
def are_collinear(point1, point2, point3, round_thres):
    """Checks if three points are collinear within a given rounding threshold.

    Args:
        point1 (tuple): The coordinates of the first point in the format (x1, y1).
        point2 (tuple): The coordinates of the second point in the format (x2, y2).
        point3 (tuple): The coordinates of the third point in the format (x3, y3).
        round_thres (int): The rounding threshold used for comparing slopes.

    Returns:
        bool: True if the points are collinear within the rounding threshold, False otherwise.
    """
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3

    # Calculate the slopes between the points
    slope_1_2 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
    slope_1_3 = (y3 - y1) / (x3 - x1) if (x3 - x1) != 0 else float('inf')
    slope_2_3 = (y3 - y2) / (x3 - x2) if (x3 - x2) != 0 else float('inf')

    # Check if the slopes are equal (or if two slopes are infinite, which means the points are vertical)
    return round(slope_1_2, round_thres) == round(slope_1_3, round_thres) == round(slope_2_3, round_thres)

# ===========================================================================================================================
def insert_point_to_polygon(point, polygon_points):
    """Inserts a point into a polygon if the point lies on an edge.

    Args:
        point (tuple, list): The coordinates of the point to be inserted in the format (x, y).
        polygon_points (list): A list of tuples representing the vertices of the polygon [(x1, y1), (x2, y2), ...].

    Returns:
        list: A new list of tuples representing the vertices of the updated polygon after inserting the point.
            If the point does not lie on an edge, the original polygon points are returned.
    """
    new_polygon_points = []
    for index in range(-1, len(polygon_points) - 1):
        if are_collinear(polygon_points[index], polygon_points[index + 1], point, 3):
            new_polygon_points += polygon_points[:index + 1]
            new_polygon_points.append(list(point))
            new_polygon_points += polygon_points[index + 1:]
            return new_polygon_points
    return polygon_points

# ===========================================================================================================================
def remove_out_of_image_polygon_points(image_width, image_hight, polygon_points):
    """
    Removes the points of the polygon that are outside the image.

    Args:
        image_width (float): The width of the image.
        image_height (float): The height of the image.
        polygon_points (list): A list of tuples representing the vertices of the polygon [(x1, y1), (x2, y2), ...].

    Returns:
        list: A new list of tuples representing the vertices of the updated polygon after removing the points outside the image.
    """
    image_lines = [[(0,0), (0, image_hight)], 
                   [(0,0), (image_width, 0)], 
                   [(0, image_hight), (image_width, image_hight)], 
                   [(image_width, 0), (image_width, image_hight)]]
    # check the intersection with image
    new_polygon = polygon_points
    for line in image_lines:
        intersection_point_list = line_polygon_intersection(line[0], line[1], polygon_points)
        if intersection_point_list:
            for intersection_point in intersection_point_list:
                new_polygon = insert_point_to_polygon(intersection_point, new_polygon)
    for point in new_polygon[::]:
        if (not 0 <= point[0] <= image_width) or (not 0 <= point[1] <= image_hight):
            new_polygon.remove(point)
    return new_polygon

# ===========================================================================================================================
def remove_out_of_image_line_points(image_width, image_hight, line_points):
    """
    Removes the points of the line that are outside the image.

    Args:
        image_width (float): The width of the image.
        image_height (float): The height of the image.
        line_points (list): A list of tuples representing the two points of the line [(x1, y1), (x2, y2)].

    Returns:
        list: A new list of tuples representing the updated line after removing the points outside the image.
    """
    image_lines = [[(0,0), (0, image_hight)], 
                   [(0,0), (image_width, 0)], 
                   [(0, image_hight), (image_width, image_hight)], 
                   [(image_width, 0), (image_width, image_hight)]]
    # check the intersection with image
    new_line = line_points
    for line in image_lines:
        intersection_point = lines_intersection(line[0], line[1], line_points[0], line_points[1])
        if intersection_point:
            new_line += [list(intersection_point)]
    for point in new_line[::]:
        if (not 0 <= point[0] <= image_width) or (not 0 <= point[1] <= image_hight):
            new_line.remove(point)
    return new_line

# ===========================================================================================================================
def rotate_points(matrix, angle_degrees):
    """
    Rotate an mx2 matrix with a 2x2 rotation matrix.

    Args:
        matrix (numpy array): The matrix of points with shape (m, 2).
        angle_degrees (numpy array): The rotation angle in degrees.

    Returns:
        numpy array: The rotated points with the same shape as the input matrix.
    """
    # Convert angle_degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Create a 2x2 rotation matrix
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                [np.sin(angle_radians), np.cos(angle_radians)]])

    # Transpose the matrix to match the dimensions for matrix multiplication
    transposed_matrix = matrix.T

    # Perform matrix multiplication to get the transformed points
    transformed_matrix = np.dot(rotation_matrix, transposed_matrix)

    # Transpose back to restore the original shape
    transformed_matrix = transformed_matrix.T

    return transformed_matrix

# ===========================================================================================================================
def rotate_image(image, angle_degrees, label_data=None):
    """
    Rotate an image by a given angle in degrees and update the associated label data.

    Args:
        image (numpy array): The input image.
        angle_degrees (float): The rotation angle in degrees.
        label_data (dict, optional): The label data associated with the image. Defaults to None.

    Returns:
        tuple: A tuple containing the rotated image and the updated label data (if provided).
    """
    # finding the cneter of image
    image_center = tuple(np.array(image.shape[1::-1]) / 2)

    # rotate image
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle_degrees, 1.0)
    final_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    # update label file if it exist
    if label_data is not None:
        # Update the data
        image_width, image_hight = label_data['imageWidth'], label_data['imageHeight']
        if angle_degrees == 90:
            for shape_index, shape in enumerate(label_data['shapes']):
                for point in shape['points']:
                    x, y = point
                    point[0] = image_hight - y
                    point[1] = x
                label_data['shapes'][shape_index] = shape
        elif angle_degrees == 180:
            for shape_index, shape in enumerate(label_data['shapes']):
                for point in shape['points']:
                    x, y = point
                    point[0] = image_width - x
                    point[1] = image_hight - y
                label_data['shapes'][shape_index] = shape
        elif angle_degrees == 270:
            for shape_index, shape in enumerate(label_data['shapes']):
                for point in shape['points']:
                    x, y = point
                    point[0] = y
                    point[1] = image_width - x
                label_data['shapes'][shape_index] = shape
        else:
            remove_shape_list = []
            for shape_index, shape in enumerate(label_data['shapes']):
                if shape['shape_type'] == 'polygon':
                    array_points = np.array(shape['points'])
                    # rotate the shape around the center of image
                    rotated_points = rotate_points(array_points - [image_width/2, image_hight/2], angle_degrees) + [image_width/2, image_hight/2]
                    # remove rotated points that are out of image
                    final_points = remove_out_of_image_polygon_points(image_width, image_hight, rotated_points.tolist())
                    # update data
                    if final_points:
                        shape['points'] = final_points
                        label_data['shapes'][shape_index] = shape
                    else:
                        remove_shape_list.append(shape_index)
                elif shape['shape_type'] == 'line':
                    array_points = np.array(shape['points'])
                    # rotate the shape around the center of image
                    rotated_points = rotate_points(array_points - [image_width/2, image_hight/2], angle_degrees) + [image_width/2, image_hight/2]
                    # remove rotated points that are out of image
                    final_points = remove_out_of_image_line_points(image_width, image_hight, rotated_points.tolist())
                    # update data
                    if final_points:
                        shape['points'] = final_points
                        label_data['shapes'][shape_index] = shape
                    else:
                        remove_shape_list.append(shape_index)
                elif shape['shape_type'] == 'point':
                    array_points = np.array(shape['points'])
                    # rotate the shape around the center of image
                    rotated_points = rotate_points(array_points - [image_width/2, image_hight/2], angle_degrees) + [image_width/2, image_hight/2]
                    # update data
                    rotated_point = rotated_points.tolist()[0]
                    # remove rotated points that are out of image
                    if (not 0 <= rotated_point[0] <= image_width) or (not 0 <= rotated_point[1] <= image_hight):
                        remove_shape_list.append(shape_index)
                    else:
                        shape['points'] = [rotated_point]
                        label_data['shapes'][shape_index] = shape
            
            list_temp = label_data['shapes'][::]
            for index in remove_shape_list:
                label_data['shapes'].remove(list_temp[index])
            
    return (final_image, label_data)

# ===========================================================================================================================
def flip_image(image, label_data=None, horizontal=True):
    """
    Flip an image horizontally or vertically and update the associated label data.

    Args:
        image (numpy array): The input image.
        label_data (dict, optional): The label data associated with the image. Defaults to None.
        horizontal (bool, optional): Flag indicating whether to flip the image horizontally. Defaults to True.

    Returns:
        tuple: A tuple containing the flipped image and the updated label data (if provided).
    """
    # flip image
    flipped_image = cv2.flip(image, int(horizontal))

    if label_data is not None:
        # Update the data
        image_shape = [label_data['imageWidth'], label_data['imageHeight']]
        for shape_index, shape in enumerate(label_data['shapes']):
            array_points = np.array(shape['points'])
            # flip the shape
            distance = array_points[:, int(not horizontal)] - image_shape[int(not horizontal)]/2
            flipeped_array_points = array_points
            flipeped_array_points[:, int(not horizontal)] = image_shape[int(not horizontal)]/2 - distance
            # update data
            shape['points'] = flipeped_array_points.tolist()
            label_data['shapes'][shape_index] = shape

    return (flipped_image, label_data)
    
# ===========================================================================================================================
def add_noise(image, intensity):
    """
    Add salt-and-pepper noise to an image.

    Args:
        image (numpy array): The input image.
        intensity (float): The intensity of the noise, ranging from 0 to 100.

    Returns:
        numpy array: The noisy image.
    """
    # add noise to image
    # Define the probability of each pixel being affected by salt-and-pepper noise
    prob = intensity/100

    # Generate salt-and-pepper noise and add it to the image
    noise = np.random.rand(*image.shape)
    noisy_image = image.copy()
    noisy_image[noise < prob / 2] = 0  # Salt noise (black)
    noisy_image[noise > 1 - prob / 2] = 255  # Pepper noise (white)

    return noisy_image

# ===========================================================================================================================
def change_brightness(image, intensity):
    """
    Adjust the brightness of an image.

    Args:
        image (numpy array): The input image.
        intensity (float): The intensity of the brightness adjustment, ranging from 0 to 100.

    Returns:
        numpy array: The adjusted image.
    """
    intensity = 255 * (intensity/100)
    
    # convert from BGR color space to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Modify the V channel (brightness)
    v = cv2.add(v, intensity)

    # Merge the modified V channel back to the HSV image
    hsv = cv2.merge((h, s, v))

    # Convert the HSV image back to BGR
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return adjusted_image

# ===========================================================================================================================
def change_contrast(image, intensity):
    """
    Adjust the contrast of an image.

    Args:
        image (numpy array): The input image.
        intensity (float): The intensity of the contrast adjustment.

    Returns:
        numpy array: The adjusted image.
    """
    # Convert the image to a floating-point data type
    img_float = image.astype(np.float32)

    # Apply the contrast adjustment formula: new_pixel_value = old_pixel_value * alpha
    new_image = np.clip(img_float * intensity, 0, 255).astype(np.uint8)
    
    return new_image
    
# ===========================================================================================================================
def image_distortion(image, fx, fy, cx, cy, k1, k2, k3, p1, p2, label_data=None):
    """
    Apply radial and tangential distortion to an image and update the associated label data.

    Args:
        image (numpy array): The input image.
        fx (float): Focal length in the x direction. Typical range from 500 to 2000 pixels.
        fy (float): Focal length in the y direction. Typical range from 500 to 2000 pixels.
        cx (float): Principal point x-coordinate. Typical range from 200 to 1000 pixels.
        cy (float): Principal point y-coordinate. Typical range from 200 to 1000 pixels.
        k1 (float): Radial distortion coefficient k1. Typical range from -0.2 to 0.2.
        k2 (float): Radial distortion coefficient k2. Typical range from -0.2 to 0.2.
        k3 (float): Radial distortion coefficient k3. Typical range from -0.2 to 0.2.
        p1 (float): Tangential distortion coefficient p1. Typical range from -0.1 to 0.1.
        p2 (float): Tangential distortion coefficient p2. Typical range from -0.1 to 0.1.
        p3 (float): Tangential distortion coefficient p3.
        label_data (dict, optional): The label data associated with the image. Defaults to None.

    Returns:
        tuple: A tuple containing the undistorted image and the updated label data (if provided).
    """
    # Define the camera matrix and distortion coefficients
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])
    dist_coeffs = np.array([k1, k2, p1, p2, k3])

    # Apply the distortion on image
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

    # Apply the distortion on labels
    for shape_index, shape in enumerate(label_data['shapes']):
        if shape['shape_type'] == 'polygon':
            # Define polygon points
            points_array = np.array(shape['points'], dtype=np.int32)
            # Create an empty mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            # Fill the polygon region in the mask
            cv2.fillPoly(mask, [points_array], 255)
            # Apply the distortion on mask
            undistorted_mask = cv2.undistort(mask, camera_matrix, dist_coeffs)
            # extract mask to polygon
            new_polygon = convert_mask_to_polygon(undistorted_mask)[0].tolist()
            shape['points'] = new_polygon
            label_data['shapes'][shape_index] = shape

    return undistorted_image, label_data

# ===========================================================================================================================
def resize_image(image, new_width, new_height, label_data=None):
    """
    Resize an image and adjust the associated label data accordingly.

    Args:
        image (numpy array): The input image.
        new_width (int): The desired width of the resized image.
        new_height (int): The desired height of the resized image.
        label_data (dict, optional): The label data associated with the image. Defaults to None.

    Returns:
        tuple: A tuple containing the resized image and the updated label data (if provided).
    """
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # calculate scaling factor
    original_height, original_width, _= image.shape
    scaling_factor_height = new_height / original_height
    scaling_factor_width = new_width / original_width

    # resize lablels
    for shape_index, shape in enumerate(label_data['shapes']):
        if shape['shape_type'] == 'polygon':
            points_array = np.array(shape['points'])
            points_array[:, 0] = points_array[:, 0] * scaling_factor_width
            points_array[:, 1] = points_array[:, 1] * scaling_factor_height
            shape['points'] = points_array.tolist()
            label_data['shapes'][shape_index] = shape

    return resized_image, label_data

# ===========================================================================================================================
def change_hue(image, hue_shift):
    """
    Modify the hue channel of an image by shifting its values.

    Args:
        image (numpy array): The input image.
        hue_shift (int): The amount to shift the hue channel (in degrees).

    Returns:
        numpy array: The modified image.
    """
    # Convert image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Modify the hue channel
    # Hue shift value (in degrees)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180

    # Convert image back to BGR
    modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    return modified_image

# ===========================================================================================================================
def augmentation(image_path, output_path=None, 
                 contrast_intensity=None, hue_shift=None, 
                 resize_parameter=None, rotate_angle=None, 
                 flip_horizontal=None, noise_intensity=None,  
                 brightness_intensity=None, distortion_parameter=None, ):
    """
    Perform image augmentation operations on an image and save the augmented image and label data (if available) to the output path.

    Args:
        image_path (str): The path to the input image.
        output_path (str, optional): The path to save the augmented image and label data. Defaults to None.
        contrast_intensity (float, optional): The intensity of contrast adjustment. Defaults to None.
        hue_shift (int, optional): The amount to shift the hue channel (in degrees). Defaults to None.
        resize_parameter (tuple, optional): The parameters for resizing the image (new_width, new_height). Defaults to None.
        rotate_angle (float, optional): The angle of rotation in degrees. Defaults to None.
        flip_horizontal (bool, optional): Flag indicating whether to flip the image horizontally. Defaults to None.
        noise_intensity (float, optional): The intensity of adding noise to the image. Defaults to None.
        brightness_intensity (float, optional): The intensity of brightness adjustment. Defaults to None.
        distortion_parameter (tuple, optional): The parameters for applying image distortion (fx, fy, cx, cy, k1, k2, k3, p1, p2, p3). Defaults to None.

    Returns:
        If output_path is not provided, returns a tuple containing the augmented image and the updated label data (if available).
    """
    
    path_and_file_name, extension = os.path.splitext(image_path)
    file_name = os.path.basename(path_and_file_name)
    path = os.path.dirname(path_and_file_name)
    new_file_name = f'{file_name}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}'

    # reading image
    image = cv2.imread(image_path)

    # reading labelMe json file if it exists
    label_data = None
    if os.path.exists(os.path.join(path, file_name + '.json')):
        # Read JSON data from the label file
        with open(os.path.join(path, file_name + '.json'), 'r') as file:
            label_data = json.load(file)
            # update new image name
            label_data['imagePath'] = new_file_name + extension

    if rotate_angle != None:
        image, label_data = rotate_image(image, rotate_angle, label_data)
    
    if flip_horizontal != None:
        image, label_data = flip_image(image, label_data, flip_horizontal)

    if noise_intensity != None:
        image = add_noise(image, noise_intensity)

    if brightness_intensity != None:
        image = change_brightness(image, brightness_intensity)

    if contrast_intensity != None:
        image = change_contrast(image, contrast_intensity)
        
    if distortion_parameter != None:
        image, label_data = image_distortion(image, *distortion_parameter, label_data)

    if resize_parameter != None:
        image, label_data = resize_image(image, 300, 2000, label_data)

    if hue_shift != None:
        image = change_hue(image, hue_shift)

    if output_path:
        # write image
        cv2.imwrite(os.path.join(output_path, new_file_name + extension), image)
        # Write the labelMe json file
        with open(os.path.join(output_path, new_file_name + '.json'), 'w') as file:
            json.dump(label_data, file, indent=4)
    else:
        return image, label_data