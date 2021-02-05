import os
import shutil
import datetime

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from cv2 import cv2

from mtcnn import MTCNN

from keras.models import load_model
from keras.preprocessing.image import img_to_array

from sklearn.manifold import TSNE


MIN_FACE_SIZE = 160



def delete_create_dirs(dirs: list):
    """Delete and re-create directories in the list of directories provided"""
    for dir in dirs:
        shutil.rmtree(dir, ignore_errors = True, onerror=None)
    for dir in dirs:
        os.mkdir(dir)


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
    #https://cristianpb.github.io/blog/image-rotation-opencv
    def get_rotated_coordinate(rot_mat, old_coord):
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
        save_image_folder,
        confidence_filter = 0.98,
        face_height_filter = MIN_FACE_SIZE,
        nose_shift_filter = 25,
        eye_line_angle_filter = 45,
        sharpness_filter = 20,
    ):

    _, file_name = os.path.split(image_path)
    image_name, img_ext = os.path.splitext(file_name)  
    
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
                        imagefile_path = save_image_folder +'\\'+ image_name + '_' + str(image_idx) + img_ext
                        cv2.imwrite(imagefile_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                    else:
                        print('image sharpness < ', sharpness_filter)
                else:
                    print('grayscale image')
            else:
                print('nose shift:', nose_shift, '>', 'filter:', nose_shift_filter, 'or eye_line_angle:', abs(angle), '>', 'filter', eye_line_angle_filter)
        else:
            print('low_confidence:', confidence, 'or height:', height, '<', 'height_filter:', face_height_filter)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def mtcnn_filter_save(directory, save_folder):    
    paths = listdir_fullpath(directory)    
    for path in paths:
        # check if this file already in savefolder:
        files_in_save = os.listdir(save_folder)
        filenames_in_save = [os.path.splitext(x)[0] for x in files_in_save]
        original_filenames = set([x[0:x.rfind('_')] for x in filenames_in_save])

        file_ = os.path.split(path)[1]
        file_name = os.path.splitext(file_)[0]

        if file_name not in original_filenames:
            #print('filename:', file_name, 'not in:', original_filenames)
            mtcnn_filter_save_single(image_path=path,
                                 save_image_folder=save_folder)



model = load_model(r'facenet_keras_pretrained/model/facenet_keras.h5')
def get_facenet_embedding(image_path, model=model):
    """Generate FaceNet embeddings
    
    Args:
        image_path (path)
    
    Returns:
        embedding (128 dimension vector)
    """

    def standardize_image(image):
        """Returns standardized image: (x-mean(X))/std(X)
        
        Args:
            image:
        
        Returns:
            image (standardized)
        """   
        mean, std = image.mean(), image.std()
        return (image - mean) / std   

    image = read_image(image_path)
    image = cv2.resize(image, (160,160))
    image=standardize_image(image)
    input_arr = img_to_array(image) # Convert single image to a batch.
    input_arr = np.array([input_arr])
    return model.predict(input_arr)[0,:]


def get_facenet_embeddings(directory):
    # Get embeddings for all    
    paths = listdir_fullpath(directory)    
    embeddings = []
    for file_path in paths:
        emb = get_facenet_embedding(file_path, model=model)
        embeddings.append(emb)
    embeddings = np.array(embeddings)
    
    return embeddings


def tsne(directory, save_directory):

    paths =  listdir_fullpath(directory)
    if len(paths) < 2:
        print('At least 2 photos required')
        pass
    else:
        X = get_facenet_embeddings(directory)
        X_tsne = TSNE(perplexity=2, learning_rate = 1000, n_iter=1000, random_state=0).fit_transform(X)

        x = X_tsne[:,0]
        y = X_tsne[:,1]

        # The idea on how to plot faces is taken from https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
        def getImage(path, size):
            image = plt.imread(path)
            image = resize_image(image, size)
            return OffsetImage(image)

        plt.rcParams["figure.figsize"] = (20,20)
        # This part is plotting faces
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        #ax.set_axis_off()
        for x0, y0, path in zip(x, y, paths):
            ab = AnnotationBbox(getImage(path, (50,50)), (x0, y0), frameon=False)
            ax.add_artist(ab)
        
        time_now  = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        # clean the folder
        for f in os.listdir(save_directory):
            os.remove(os.path.join(save_directory, f))
        #print('save_dir:', save_directory)  

        fig.savefig(save_directory + '/' + time_now + 'tsne.png') # changing name is usefull so that browser cashing is avoided

        # This part is plotting colors
        # creation_dates = df_filtered.creation_date
        # Creation date is shown with colors on the plot below
        # Spectral(rainbow) palette is used with older photos shown in red and recent in blue
        
        #sns.scatterplot(x=x, y=y, hue = creation_dates, s=7000, palette=sns.color_palette('Spectral',len(set(creation_dates))))
        
        #plt.legend([],[], frameon=False) # hide the legend if it too long

        # Set figure background
        #sns.set_style("whitegrid", {'axes.grid' : False,'axes.facecolor': 'white'})


#if __name__ == '__main__':
    #image = read_image(image_path)
    #plt.imshow(image)
    #plt.show()

    #image_path = 'Test/Photoset/2008/1.jpg'
    #mtcnn_filter_save_single(image_path)
    
    #directory = 'Test/Photoset/2008'
    #mtcnn_filter_save(directory)