import argparse
import os
import glob

from torch.utils.data import Dataset
from torchvision.io import read_image

class BackDataset(Dataset):
    def __init__(self,root,list_of_dirs,img_transform = None,labels_transform=None):
        self.filenames = []
        self.filepaths = {}
        for pt_dir in list_of_dirs:
            img_dir = os.path.join(root,pt_dir,"images")
            labels_dir = os.path.join(root,pt_dir,"labels")
            for file in glob.glob(os.path.join(img_dir,"*.jpg")):
                filename = file.split("/")[-1][:-4]
                self.filenames.append(filename)
                self.filepaths[filename] = [file,os.path.join(labels_dir,filename + ".png")]



        self.img_transform = img_transform
        self.labels_transform = labels_transform

    def __len__(self):
        return(len(self.filenames))
    def __getitem__(self, index):
        paths = self.filepaths[self.filenames[index]]
        image = read_image(paths[0])/255
        label = read_image(paths[1])>0
        return({"image":image, "mask":label})
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage='use "%(prog)s --help" for more information',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("root",help="used input directory for the BackDataset." +'\n' + "The directory must use this architecture:\nroot\n |--- train-val\n |      |--- annotated...\n |      |      |--- images (*.jpg)\n |      |      |---labels (*.png)\n |      |    ...\n |      |--- annotated...\n |--- test\n |      |--- annotated...")
    parser.add_argument("train-val ratio",help="Ratio (between 0 and 1) of patients in train-val directory to be used for training",type=float,default=0.8)
    args = parser.parse_args()