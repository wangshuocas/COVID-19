"""
"""

from generator import Generator

import os
import h5py
import json
import scipy.io as sio
import numpy as np
from six import raise_from
from PIL import Image


lungNCP_classes = {
    'normal'    : 0,
    'NCP'       : 1
}


class LungNCPGenerator(Generator):

    def __init__(
        self,
        data_dir,
        classes=lungNCP_classes,
        **kwargs
    ):
        self.data_dir             = data_dir
        self.classes              = classes

        self.image_names = []
        f = open(data_dir)
        while True:
            l = f.readline()
            if not l:
                break
            self.image_names.append(l.strip())
        f.close()
            
        self.labels = {}#{1:'COVID-19'}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(LungNCPGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return len(self.classes)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def _read1image(self, image_index):
        f1Dir = self.image_names[image_index]
        ID2 = f1Dir.split('/')[-2]
        ID1 = f1Dir.split('/')[-3]#center
        ID = ID1 + '_' + ID2
        NCPLabel = int(f1Dir.split('/')[-1].split('_')[-2][1])
        center = f1Dir.split('/')[-3]
        
        img_data = sio.loadmat(f1Dir)
        img = img_data['only_lung_zoomed_3std']
        image = img.transpose(2,0,1)
        
        return ID, image, NCPLabel, center

    def load_image_annotations(self, image_index):
        ID, image, NCPlabel, center = self._read1image(image_index)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=-1)
        annotations = {}
        annotations['ID'] = ID
        annotations['NCPLabel'] = NCPlabel
        
        return image, annotations
    
    def compute_targets(self, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        IDBatch = np.asarray(['d'*100]*len(annotations_group))
        NCPLabelBatch = np.asarray([-1]*len(annotations_group))
        #zhongzhengnowBatch = np.asarray([[-1.0, -1]]*len(annotations_group))
        #zhongzheng1weekBatch = np.asarray([[-1.0, -1]]*len(annotations_group))
        for idx, ag in enumerate(annotations_group):
            IDBatch[idx] = ag['ID']
            NCPLabelBatch[idx] = ag['NCPLabel']
            #zhongzhengnowBatch[idx] = ag['zhongzhengnow']
            #zhongzheng1weekBatch[idx] = ag['zhongzheng1week']
        if self.saveProbFlag == False:
            return NCPLabelBatch
        else:
            return (NCPLabelBatch, IDBatch)
    
    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        inputs, targets = self.compute_input_output(group)
        #return inputs, tuple(targets[:2])# for testing phase
        return inputs, targets#for training phase
    
    
    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        assert(len(image_group) == len(annotations_group))
        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index] = (image_group[index] - image_group[index].mean()) / image_group[index].std()

        return image_group, annotations_group
    
    
