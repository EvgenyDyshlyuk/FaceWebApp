import os
import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np

MIN_FACE_SIZE = 10

def read_image(file_path):
    """Read jpg image file using cv2

    Args:
        file_path (path): path to the file

    Returns:
        image: 
    """
    imageBGR = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR) # Use to open unicode folders/files. https://stackoverflow.com/questions/43185605/how-do-i-read-an-image-from-a-path-with-unicode-characters
    image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB) # cv2 uses BGR color ordering, so need to change order for RGB. https://stackoverflow.com/questions/52494592/wrong-colours-with-cv2-imdecode-python-opencv
    return image

detector = MTCNN(min_face_size = MIN_FACE_SIZE)

def get_MTCNN_result(file_path):
    """Get mtcnn result for 1 jpg file
    
    Args:
        file_path (path): file path

    Returns:
        MTCNN result (list):
    """
    image = read_image(file_path)
    return detector.detect_faces(image) 

def variance_of_laplacian(image):
    """Assessment of bluriness of an image. Bigger value - lower bluriness.
    Idea taken from https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    and modified. I found it emperically that (max-min)**2/var is working better than simple variance
    as an assessment of bluriness. 

    Args:
        image
        
    Returns:
        assessment of bluriness
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    conv2d = cv2.Laplacian(gray_image, cv2.CV_64F)    
    return int((conv2d.max() - conv2d.min())**2/(2*conv2d.var()))

def get_rotated_keypoints(keypoints, rot_mat):
    """Rotate coordinates of keypoints

    Args:
        keypoints (dict)
        rot_mat: rotation matrix
    
    Returns:
        keypoints in new coordinates
    """    
    def get_rotated_coordinate(rot_mat, old_coord): #https://cristianpb.github.io/blog/image-rotation-opencv
        expanded_coord = old_coord + (1,) 
        product = np.dot(rot_mat,expanded_coord)
        new_coord = int(product[0]), int(product[1])               
        return new_coord
    
    keypoints_rotated = keypoints.copy()
    for key, value in keypoints_rotated.items():
        keypoints_rotated[key] = get_rotated_coordinate(rot_mat, value)
    return keypoints_rotated

def is_not_grayscale(image):
    """Check if image is grayscale

    Args:
        image
    
    Returns:
        Bool: True if not grayscale
    """ 
    return image[:, :, 0].sum() != image[:, :, 1].sum()

def resize_image(image, new_size):
    """Resize image

    Args:
        image
        new_size (int,int): new size of an image
    
    Returns:
        image: rescaled image
    """ 
    image_resized = cv2.resize(image, new_size)
    return image_resized


def mtcnn_filter_save_single(
        image_path,
        save_image_folder = 'cropped',
        confidence_filter = 0.98,
        face_height_filter = 100,
        nose_shift_filter = 25,
        eye_line_angle_filter = 45,
        sharpness_filter = 20,
    ):

    _, file_name = os.path.split(image_path)
    image_name, img_ext = file_name.split('.')
    print(image_name, img_ext)

    MTCNN_res = get_MTCNN_result(image_path)

    for image_idx, res in enumerate(MTCNN_res):
        confidence = res['confidence']
        bounding_box = res['box']
        keypoints = res['keypoints']
        upper_left_x, upper_left_y, width, height  = bounding_box

        # change box from rectangular to square
        side = max(height, width)
        upper_left_x = int(upper_left_x + width/2 - side/2)
        width=height=side

        if (confidence >= confidence_filter) and (height >= face_height_filter):
            # find an angle of line of eyes.
            dY = keypoints['right_eye'][1] - keypoints['left_eye'][1]
            dX = keypoints['right_eye'][0] - keypoints['left_eye'][0]
            angle = np.degrees(np.arctan2(dY,dX))
            # calculate rotation matrix for this anlge around nose as a central point 
            rot_mat = cv2.getRotationMatrix2D(keypoints['nose'], angle, 1.0)
            # calculate new coordinates of keypoints
            keypoints_rotated = get_rotated_keypoints(keypoints, rot_mat)
            # calculate nose shift
            nose_shift = 100*abs((keypoints_rotated['nose'][0] - keypoints_rotated['left_eye'][0] - dX/2)/dX)

            if (nose_shift <= nose_shift_filter) and (abs(angle) <= eye_line_angle_filter):
                image = read_image(image_path)

                if is_not_grayscale(image):
                    image_rotated = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
                    image_cropped_central_part = image_rotated[int(upper_left_y+height*1/2):int(upper_left_y + height*2/3), int(upper_left_x+width*1/3):int(upper_left_x + width*2/3)]
                    sharpness = variance_of_laplacian(image_cropped_central_part)

                    if sharpness >= sharpness_filter:
                        image_cropped = image_rotated[upper_left_y:upper_left_y + height, upper_left_x:upper_left_x + width]
                        image_resized = resize_image(image_cropped, (160,160))
                        imagefile_path = save_image_folder +'\\'+ image_name + '_' + str(image_idx) + '.' + img_ext
                        cv2.imwrite(imagefile_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))

def mtcnn_filter_save(directory):
    
    def listdir_fullpath(d):
        return [os.path.join(d, f) for f in os.listdir(d)]
    
    paths = listdir_fullpath(directory)
    
    for path in paths:
        mtcnn_filter_save_single(image_path=path)


if __name__ == '__main__':
    #image = read_image(image_path)
    #plt.imshow(image)
    #plt.show()

    #image_path = 'Test/Photoset/2008/1.jpg'
    #mtcnn_filter_save_single(image_path)
    
    directory = 'Test/Photoset/2008'
    mtcnn_filter_save(directory)


