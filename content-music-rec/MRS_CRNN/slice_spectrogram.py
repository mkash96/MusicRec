import os
import re
from PIL import Image

"""
Slice the spectrogram into multiple 128x128 images which will be the input to the
Convolutional Neural Network.
"""
def slice_spect(verbose=0, mode=None):
    if mode=="Train":
        if os.path.exists('Train_Sliced_Images'):
            return
        labels = []
        image_folder = "Train_Spectogram_Images"
        filenames = [os.path.join(image_folder, f) for f in os.listdir(image_folder)    #A list of all .jpg files in the Train_Spectogram_Images directory.
                       if f.endswith(".jpg")]
        counter = 0     #A variable used to keep track of the number of slices generated.
        if(verbose > 0):
            print ("Slicing Spectograms ...")
        if not os.path.exists('Train_Sliced_Images'):
            os.makedirs('Train_Sliced_Images')
        for f in filenames:
            genre_variable = re.search('Train_Spectogram_Images/.*_(.+?).jpg', f).group(1)      #extract the genre label from the filename. The filenames are expected to have the format {counter}_{genre}.jpg
            img = Image.open(f)     #Loads the spectrogram image using PIL.
            subsample_size = 128    #Defines the size of each slice as 128 pixels
            width, height = img.size
            number_of_samples = int(width / subsample_size)     #Determines how many 128-pixel-wide slices can be made from the image by dividing the width by the subsample size.
            for i in range(number_of_samples):
                start = i*subsample_size        #Calculate Start Position
                img_temporary = img.crop((start, 0., start + subsample_size, subsample_size))       #Uses img.crop() to extract a 128x128 pixel region from the original image.
                img_temporary.save("Train_Sliced_Images/"+str(counter)+"_"+genre_variable+".jpg")   #Saves the cropped image to the Train_Sliced_Images directory with a filename that includes the counter and genre label.
                counter = counter + 1
        return

    elif mode=="Test":
        if os.path.exists('Test_Sliced_Images'):
            return
        labels = []
        image_folder = "Test_Spectogram_Images"
        filenames = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                       if f.endswith(".jpg")]
        counter = 0
        if(verbose > 0):
            print ("Slicing Spectograms ...")
        if not os.path.exists('Test_Sliced_Images'):
            os.makedirs('Test_Sliced_Images')
        for f in filenames:
            song_variable = re.search('Test_Spectogram_Images/(.+?).jpg', f).group(1)
            img = Image.open(f)
            subsample_size = 128
            width, height = img.size
            number_of_samples = int(width / subsample_size)
            for i in range(number_of_samples):
                start = i*subsample_size
                img_temporary = img.crop((start, 0., start + subsample_size, subsample_size))
                img_temporary.save("Test_Sliced_Images/"+str(counter)+"_"+song_variable+".jpg")
                counter = counter + 1
        return
