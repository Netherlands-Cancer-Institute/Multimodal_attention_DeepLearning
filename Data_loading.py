import numpy as np
from skimage.io import imread

def read_mg(file_paths, img_rows, img_cols, as_gray, channels):
    """
  Read the image files (mammography) and normalize the pixel values
    @params:
      file_paths - Array of file paths to read from
      img_rows - The image height.
      img_cols - The image width.
      as_grey - Read the image as Greyscale.
      channels - Number of channels.       
    """
    images=[]
 
    for file_path in file_paths:
        images.append(imread(file_path,  as_gray))  
        
    images = np.asarray(images, dtype=np.float32)
    images = np.stack((images,)*3, axis=-1)
    images = (images-images.min())/(images.max()-images.min())
    images = images.reshape(images.shape[0], img_rows, img_cols, channels)
    return images
    
    
def read_us(file_paths, img_rows, img_cols, as_gray, channels):
    """
  Reads the image files (ultrasound) and normalize the pixel values
    @params:
      file_paths - Array of file paths to read from
      img_rows - The image height.
      img_cols - The image width.
      as_grey - Read the image as Greyscale.
      channels - Number of channels.   
    """
    images=[]
  
    for file_path in file_paths:
      
        images.append(imread(file_path,  as_gray))
  
    images = np.asarray(images, dtype=np.float32)
    images = np.stack((images,)*3, axis=-1)
    images = (images-images.min())/(images.max()-images.min())
    images = images.reshape(images.shape[0], img_rows, img_cols, channels)
    return images
