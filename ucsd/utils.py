import numpy as np
from PIL import Image

from os import listdir
from os.path import isfile, join, isdir

TRAIN_PATH = './UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train'
TEST_PATH  = './UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test032'

def get_clips_by_stride(stride, frames_list, sequence_size):
    """ For data augmenting purposes.
    Parameters
    ----------
    stride : int
        The desired distance between two consecutive frames
    frames_list : list
        A list of sorted frames of shape 256 X 256
    sequence_size: int
        The size of the desired LSTM sequence
    Returns
    -------
    list
        A list of clips , 10 frames each
    """
    clips = []
    sz = len(frames_list)
    clip = np.zeros(shape=(sequence_size, 256, 256, 1))
    cnt = 0
    for start in range(0, stride):
        for i in range(start, sz, stride):
            clip[cnt, :, :, 0] = frames_list[i]
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(clip)
                cnt = 0
    return clips


def get_train_set():
    """
        return shape of (NUMBER_OF_SEQUENCES,SINGLE_SEQUENCE_SIZE,FRAME_WIDTH,FRAME_HEIGHT,1)
    """
    clips = []
    
    for f in sorted(listdir(TRAIN_PATH)):   # loop over each folder
        if isdir(join(TRAIN_PATH, f)):
    #         print(f)
            all_frames = []

            for c in sorted(listdir(join(TRAIN_PATH, f))): # loop over each image within the current folder
                if str(join(join(TRAIN_PATH, f), c))[-3:] == "tif":
    #                 print(c)
                    img = Image.open(join(join(TRAIN_PATH, f), c)).resize((256, 256))
                    img = np.array(img, dtype=np.float32) / 256.0
                    all_frames.append(img)
            
            # get the 10-frames sequences from the list of images after applying data augmentation
            for stride in range(1, 3):
                clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=10))
                
    return np.array(clips)


def get_single_test():
    sz = 200
    test = np.zeros(shape=(sz, 256, 256, 1))
    cnt = 0
    for f in sorted(listdir(TEST_PATH)):
        if str(join(TEST_PATH, f))[-3:] == "tif":
            img = Image.open(join(TEST_PATH, f)).resize((256, 256))
            img = np.array(img, dtype=np.float32) / 256.0
            test[cnt, :, :, 0] = img
            cnt = cnt + 1
    return test