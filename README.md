# Image Augmentation

This code provides a function for performing image augmentation operations on an image. It supports various augmentation techniques such as contrast adjustment, hue shift, resizing, rotation, flipping, noise addition, brightness adjustment, and image distortion. The augmented image and annotation file (labelme json annotation if available) can be saved to an output path.

## Prerequisites

Before using this code, make sure you have the following:

- Python 3 installed on your system

- The required Python packages installed. You can install them by running the following command:

  ```shell
  pip install -r requirements.txt

## Usage

```python
def augmentation(image_path, output_path=None, 
                 contrast_intensity=None, hue_shift=None, 
                 resize_parameter=None, rotate_angle=None, 
                 flip_horizontal=None, noise_intensity=None,  
                 brightness_intensity=None,
                 distortion_parameter=None):
```

- `image_path` (str): The path to the input image.
- `output_path` (str, optional): The path to save the augmented image and label data. If not provided, the augmented image and label data will be returned as a tuple. Defaults to `None`.
- `contrast_intensity` (float, optional): The intensity of contrast adjustment. Defaults to `None`.
- `hue_shift` (int, optional): The amount to shift the hue channel (in degrees). Defaults to `None`.
- `resize_parameter` (tuple, optional): The parameters for resizing the image (`new_width`, `new_height`). Defaults to `None`.
- `rotate_angle` (float, optional): The angle of rotation in degrees. Defaults to `None`.
- `flip_horizontal` (bool, optional): Flag indicating whether to flip the image horizontally. Defaults to `None`.
- `noise_intensity` (float, optional): The intensity of adding noise to the image. Defaults to `None`.
- `brightness_intensity` (float, optional): The intensity of brightness adjustment. Defaults to `None`.
- `distortion_parameter` (tuple, optional): The parameters for applying image distortion (`fx`, `fy`, `cx`, `cy`, `k1`, `k2`, `k3`, `p1`, `p2`, `p3`). Defaults to `None`.

To perform image augmentation, call the augmentation function and provide the necessary arguments based on the desired augmentation operations. The augmented image and label data (if available) can be saved to an output path by specifying the output_path argument. If output_path is not provided, the function will return a tuple containing the augmented image and the updated label data (if available).

## Example

```python
image_path = 'path/to/input/image.jpg'
output_path = 'path/to/save/augmented/image/'

# Perform augmentation with specific parameters
augmentation(image_path, output_path=output_path, 
             contrast_intensity=0.5, hue_shift=30, 
             resize_parameter=(800, 600), rotate_angle=45, 
             flip_horizontal=True, noise_intensity=0.2,  
             brightness_intensity=-0.3, distortion_parameter=(0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
```

This example performs multiple augmentation operations on the input image and saves the augmented image and annotation file (if available) to the specified output path.
