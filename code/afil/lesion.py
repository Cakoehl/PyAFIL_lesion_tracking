#! env python3

# -*- coding: utf-8 -*-

__author__ = "Caroline Köhlere"
__email__ = "caroline.koehler@uniklinikum-dresden.de"
__license__ = "BSD 3 Clause"
__version__ = "0.1"

import os
from nilearn.plotting import plot_anat, view_img, plot_roi
import matplotlib as plt
from bids.layout import BIDSLayout
from os.path import abspath, dirname, join, exists
import nibabel as nib
import numpy as np
from skimage.measure import label
from skimage import io
from nilearn.image import image  # from nilearn.plotting import view_img
import scipy as sp
from tqdm import tqdm
import pandas as pd



# class Lesion---------------------------------------------------------------------
class Lesion:
    """ 
    creates a Lesion object
    
    """

    def __init__(self, session, label):



class ILesion(Lesion):
    """
    creates a Lesion object

    """

    def __init__(self, label, session):
        self.session = session
        self.label = label
        self._glabel = None
        self.parents = []
        self.childs = []
        self.volume = None
        self.new = False

    def __repr__(self):
        return str(self.label)

    def __repr__(self):
        return str(self._glabel)

    def get_new_lesions(self, interim_label_list_fu, intersect_fu):
        """
        find new lesion label in follow-up time point
        this is True when label in interim_label_list_fu () is not member of intersect_fu list
        (label has no intersection with prior lesions)
        :param interim_label_list_fu: list
        :param intersect_fu:
        :return: bool
        """
        # intersecting labels of [prior interim label, follow up interim label]
        intersect_fu = np.array(intersect_fu)
        # array of follow-up labels which have a prior lesion
        intersect_fu_labels = intersect_fu[:, 1]
        # find new lesions in label list with no prior intersection
        self.new_lesions = np.setxor1d(interim_label_list_fu, intersect_fu_labels, assume_unique=True)
        # print('new lesions', self.new_lesions)
        if np.isin(self.new_lesions, self.label, assume_unique=True).any():
            # suspicious new lesion, in fu, (but to do check if it is ture for all prior time-points)
            self.sus_new = True
        else:
            self.sus_new = False


    def get_all_descendents(self):
        """
        get all interim fu labels of spezified session label, ask all childs for there
        child and intern recall get_all_descendents() until no more child can be found, --> get all interim follow-up labels of spezified interim label
        Returns:
        desc: list of interim fu labels (all child, grand-son, great-grand-son etc. )
        """
        descs = []
        for child in self.childs:
            descs.append(child)
            descs.append(child.get_all_descendents())
            # print(f'descs of {self.session.session_id}({self.label})', descs)
        return descs



    def get_all_ancestor(self):
        """
        get all interim ancestor labels of spezified session label (get all prior labels)
        Returns:
        desc: list of interim ancestor labels

        """
        ancs = []
        for parent in self.parents:
            ancs.append(parent)
            ancs.append(parent.get_all_ancestor())
            print(f'ancestor of sesssion (ilbl) {self.session.session_id}({self.label})', ancs)

        return ancs


    def add_child(self, child_lesion):
        """
        add child lesion label in fu-time-point if lesion get separated (lesion has more than 1 child)
        :param child_lesion:
        :return:
        """
        # print(f"Trying to ad child {child_lesion.label} for Lesion {self.label}")
        print('child_lesion', child_lesion, type(child_lesion))
        if type(child_lesion) is Lesion:
            if not child_lesion in self.childs:
                self.childs.append(child_lesion)
                child_lesion.add_parent(self)
                print(child_lesion.add_parent(self))
        else:
            raise ValueError("not a valid Lesion type")

            # a new lesion was observed

    def add_parent(self, parent_lesion):
        if type(parent_lesion) is Lesion:
            if not parent_lesion in self.parents:
                self.parents.append(parent_lesion)
                parent_lesion.add_child(self)

        else:
            raise ValueError("not a valid Lesion type")

    def get_related_glabel(self, globl_intersect):
        """
        Lesion object get the global label from glbl_intersect of session object
        """
        glabel_index = np.where(globl_intersect[:, 1] == self.label)
        self._glabel = globl_intersect[:, 0][glabel_index][0]
        return self._glabel

    @property
    def glabel(self):
        if hasattr(self, '_glabel'):
            return self._glabel
        else:
            raise ValueError('No global label. Uses calc_intersect before')

    def get_volume(self, lbl_mask):
        """
        get lesion volume of specified label

        Parameters:
        -----------
        lbl_mask: np.array
            labeled global or interim lesion mask

        Returns:
        --------

        volume: int

        """
        self.volume = (lbl_mask[lbl_mask == self.label] > 0).astype(np.int).sum()

    def lesion_describe(self, lbl_mask, maps):
        """
        This function masks the lesion area in the quantitative map and calculates
        some lesion statistics based on provided stat_func().

        Args:
            lbl_mask: numpy.ndarray labeled lesion mask (ilbl or glbl)
            maps: str which is the filename of the quantitative map

        Returns:  self.l_stats: dict , lesion statistics
                  l_stats_df , a DataFrame consisting of one row index == label, and lesion stats
        Example:
            l_stats_df
         	mean 	    std 	    median 	    1_quantile 	3_quantile 	min 	    max       map
        2 	0.248093 	0.004892 	0.247099 	0.245824 	0.252232 	0.239431 	0.256164  'mwf'

        """
        self.l_stats = {}
        self.l_stats_df = pd.DataFrame()
        try:
            for map_key in maps:
                qmap = maps[map_key]
                qmap = nib.load(qmap).get_fdata()
                l_metric = qmap[lbl_mask == self.label]
                self.l_stats = stat_func(l_metric)
                self.l_stats['map'] = map_key
                df = pd.DataFrame(self.l_stats, index = [self.label])
                self.l_stats_df = self.l_stats_df.append(df)
            return self.l_stats_df
        except:
            print('no maps found, specify maps in .json config file')

class GLesion(Lesion):
    """ create a global lesion object of the time series
    """

    def __init__(self, glbl):
        #self.glbl_intersect_to_df = glbl_intersect_to_df
        self.glbl = glbl
        self.new_glabel = False

    def lesion_describe(self, relbl_mask, maps):
        """
            This function masks the lesion area in the quantitative map and calculates
            some lesion statistics based on provided stat_func().

            Args:
                lbl_mask: numpy.ndarray labeled lesion mask (glbl)
                maps: dict map_key=modality, value= str, filename of the quantitative map

            Returns:  self.l_stats: dict , lesion statistics
                      l_stats_df , a DataFrame consisting of one row index == label, and lesion stats
            Example:
                l_stats_df
             	mean 	    std 	    median 	    1_quantile 	3_quantile 	min 	    max       map
            2 	0.248093 	0.004892 	0.247099 	0.245824 	0.252232 	0.239431 	0.256164  'mwf'

       """

        self.l_stats = {}
        self.l_stats_df = pd.DataFrame()
        for map_key in maps:
            qmap = maps[map_key]
            qmap = nib.load(qmap).get_fdata()
            l_metric = qmap[relbl_mask == self.glbl]
            self.l_stats = stat_func(l_metric)
            self.l_stats['map'] = map_key
            df = pd.DataFrame(self.l_stats, index=[self.glbl])
            self.l_stats_df = self.l_stats_df.append(df)
        return self.l_stats_df

