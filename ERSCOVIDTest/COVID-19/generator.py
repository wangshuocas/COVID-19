"""
"""

import numpy as np
import random

import tensorflow
import tensorflow.keras as keras

from image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    resize_image,
)

class Generator(tensorflow.keras.utils.Sequence):
    """ Abstract generator class.
    """

    def __init__(
        self,
        transform_generator = None,
        visual_effect_generator=None,
        batch_size=1,
        group_method='random',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        image_min_side=800,
        image_max_side=1333,
        no_resize=False,
        transform_parameters=None,
        config=None
    ):
        self.transform_generator    = transform_generator
        self.visual_effect_generator = visual_effect_generator
        self.batch_size             = int(batch_size)
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.no_resize              = no_resize
        self.transform_parameters   = transform_parameters or TransformParameters()
        self.config                 = config
        
        if self.config.has_option('MODEL', 'saveProb'):
            self.saveProbFlag = self.config.getboolean('MODEL', 'saveProb')
        else:
            self.saveProbFlag = False

        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle_groups:
            self.group_images()
            random.shuffle(self.groups)

    def size(self):
        """ Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """ Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """ Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """ Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def image_path(self, image_index):
        """ Get the path to an image.
        """
        raise NotImplementedError('image_path method not implemented')

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')
    
    def load_image_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        raise NotImplementedError('load_image_annotations method not implemented')

    def load_annotations_group(self, group):
        """ Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        for annotations in annotations_group:
            assert(isinstance(annotations, dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(type(annotations))
            assert('labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def load_image_group(self, group):
        """ Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def random_visual_effect_group_entry(self, image, annotations):
        """ Randomly transforms image and annotation.
        """
        visual_effect = next(self.visual_effect_generator)
        # apply visual effect
        image = visual_effect(image)
        return image, annotations

    def random_visual_effect_group(self, image_group, annotations_group):
        """ Randomly apply visual effect on each image.
        """
        assert(len(image_group) == len(annotations_group))

        if self.visual_effect_generator is None:
            # do nothing
            return image_group, annotations_group

        for index in range(len(image_group)):
            # apply effect on a single group entry
            image_group[index], annotations_group[index] = self.random_visual_effect_group_entry(
                image_group[index], annotations_group[index]
            )

        return image_group, annotations_group

    def random_transform_group_entry(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

        return image, annotations

    def random_transform_group(self, image_group, annotations_group):
        """ Randomly transforms each image and its annotations.
        """

        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_transform_group_entry(image_group[index], annotations_group[index])

        return image_group, annotations_group

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        if self.no_resize:
            return image, 1
        else:
            return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        raise NotImplementedError('preprocess_group_entry method not implemented')

    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        raise NotImplementedError('preprocess_group method not implemented')

    def group_images(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            print('ratio is not used ')
            #order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(4))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2], :image.shape[3]] = image

        if keras.backend.image_data_format() == 'channels_first':
            image_batch = image_batch.transpose((0, 3, 1, 2))

        return image_batch

    def compute_targets(self, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        IDBatch, EGFRLabelBatch = np.asarray(['default']*len(annotations_group)), np.asarray([-1]*len(annotations_group))
        for idx, ag in enumerate(annotations_group):
            IDBatch[idx] = ag['ID']
            EGFRLabelBatch[idx] = ag['label']
        return [EGFRLabelBatch, IDBatch]

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group, annotations_group = [], []
        for image_index in group:
            image, annotations = self.load_image_annotations(image_index)
            image_group.append(image)
            annotations_group.append(annotations)

        # randomly apply visual effect
        image_group, annotations_group = self.random_visual_effect_group(image_group, annotations_group)

        # randomly transform data
        image_group, annotations_group = self.random_transform_group(image_group, annotations_group)
        
        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)
        
        #inputs = self.compute_inputs(image_group)
        inputs = np.asarray(image_group)

        # compute network targets
        targets = self.compute_targets(annotations_group)

        return inputs, targets

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        inputs, targets = self.compute_input_output(group)

        return inputs, targets[0]
