#! env python3

# -*- coding: utf-8 -*-

__author__ = "Caroline Köhler"
__email__ = "caroline.koehler@uniklinikum-dresden.de"
__license__ = "BSD 3 Clause"
__version__ = "0.1"

import os


from os.path import  join, exists, isfile
import nibabel as nib
import numpy as np
from skimage.measure import label
from tqdm import tqdm
import pandas as pd
import ast
import json


def intersection(glbl_mask, num_sum, label_ses_mask, num_ses):
    """ determine spatial intersection of of labeled masks 
    
    Parameters:
    -----------
    glbl_mask: np.array
       labeled  lesion mask (global)
    num_sum: integer
        number of found labels
    label_ses_mask: np.array
       lesion mask of the session
    num_ses: integer
        number of found interim labels in session    

    Returns:
    --------

       glbl_intersect: list of intersecting labels of two time points [global_label1, interim_labelx]

    """
    intersect = []
    for i in range(1, num_sum + 1, 1):
        tmp = (glbl_mask == i)
        for j in range(1, num_ses + 1, 1):
            overlap = tmp & (label_ses_mask == j)
            if overlap.any():
                #lbls = np.array([i, j], dtype='int')
                lbls = [i, j]
                intersect.append(lbls)

    return intersect

def sort_func(session_list):
    """
    This function sort the list of session_ids.

    Example: session_list ['0m', , '1m', '15m', '2m']
            sorted_ses_list ['0m', '1m', '2m', '15m']

    :param session_list: list of str or int
    :return: sorted_ses_list: list of str or int
    """
    # replace m of session_id
    repl_ses_list=[int(ses_name.replace('m', '')) for ses_name in session_list ]
    # sort session_id which are now int
    repl_ses_list.sort()
    # add m to sorted session_id again
    sorted_ses_list = [f'{ses_name}m' for ses_name in repl_ses_list]
    return sorted_ses_list

def stat_func(l_metric):
    """
    calculates the lesion statistic for an individual lesion
    Args:
        l_metric: is a list of masked lesion values of a quantitative map for an individual label
         (l_metric is calculated in the lesion class)

    Returns: l_stats : dict of lesion statistic of the individual lesion

    """
    l_stats={
          'mean': [np.mean(l_metric)],
          'std' : [np.std(l_metric)],
          'median' : [np.quantile(l_metric,0.5)],
          '1_quantile' : [np.quantile(l_metric,0.25)],
          '3_quantile' : [np.quantile(l_metric,0.75)],
          'min' : [np.min(l_metric)],
          'max' : [np.max(l_metric)]
             }
    return l_stats

def get_vol(lbl, mask):
    """
    get lesion volume of specified label
    Parameters:
    -----------
    mask: np.array, labeled mask
    lbl:  int, global or interim label

    Returns:
    --------
    volume: int
    """
    volume = (mask[mask == lbl] > 0).astype(np.int32).sum()
    return volume

def lesion_describe(lbl, mask, maps):
    """
    This function masks the lesion area in the quantitative map and calculates
    some lesion statistics based on provided stat_func().
    Args:
        lbl: int , any label (ilbl or glbl)
        mask: numpy.ndarray labeled ilbl or glbl lesion mask
        maps: dict map_key=modality, value= str, filename of the quantitative map

    Returns:
        l_stats: dict , lesion statistics
        l_stats_df , a DataFrame consisting of row index == label, and lesion stats of different modality
    Example:
    l_stats_df
          mean 	    std 	    median 	    1_quantile 	3_quantile 	min 	    max       map
     2 	0.248093 	0.004892 	0.247099 	0.245824 	0.252232 	0.239431 	0.256164  'mwf'
       """

    l_stats = {}
    l_stats_df = pd.DataFrame()
    for map_key in maps:
        qmap = maps[map_key]
        qmap = nib.load(qmap).get_fdata()
        l_metric = qmap[mask == lbl]
        l_stats = stat_func(l_metric)
        l_stats['map'] = map_key
        l_stats['volume'] = (mask[mask == lbl] > 0).astype(np.int32).sum()
        df = pd.DataFrame(l_stats, index=[lbl])
        l_stats_df = pd.concat([l_stats_df, df])

    return l_stats, l_stats_df

# class Lesion---------------------------------------------------------------------
class ILesion:
    """ 
    creates a Lesion object
    
    """
    def __init__(self, label):
        self.ilbl = label
        self.glbl = None
        self.parents = []
        self.childs = []
        self.sus_new = False
        self.l_stats = {}
        self.l_stats_df= None

    def __repr__(self):
        return str(self.ilbl)

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
        new = np.setxor1d(interim_label_list_fu, intersect_fu_labels, assume_unique=True)
        # print('new lesions', self.new_lesions)
        if np.isin(new, self.ilbl, assume_unique=True).any():
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
            #print(f'descs of {self.session.session_id}({self.ilbl})', descs)
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
            #print(f'ancestor of sesssion (ilbl) {self.session.session_id}({self.ilbl})', ancs)

        return ancs

    def add_child(self, child_lesion):
        """
        add child lesion label in fu-time-point if lesion get separated (lesion has more than 1 child)
        :param child_lesion:
        :return:
        """
        # print(f"Trying to ad child {child_lesion.label} for Lesion {self.label}")
        #print('child_lesion', child_lesion, type(child_lesion))
        if type(child_lesion) is ILesion:
            if not child_lesion in self.childs:
                self.childs.append(child_lesion)
                child_lesion.add_parent(self)
                #print(child_lesion.add_parent(self))
        else:
            raise ValueError("not a valid Lesion type")


    def add_parent(self, parent_lesion):
        if type(parent_lesion) is ILesion:
            if not parent_lesion in self.parents:
                self.parents.append(parent_lesion)
                parent_lesion.add_child(self)
        else:
            raise ValueError("not a valid Lesion type")

    def get_related_glabel(self, glbl_intersect):
        """
        Lesion object get the global label from glbl_intersect of session object
        """
        glabel_index = np.where(glbl_intersect[:, 1] == self.ilbl)
        self.glbl = glbl_intersect[:, 0][glabel_index][0]

    def ilbl_describe(self, ilbl_mask, maps):

        """
        This function masks the lesion area in the quantitative map and calculates
               some lesion statistics based on provided stat_func().#

        Args:
            ilbl_mask: numpy.ndarray labeled lesion mask (ilbl or glbl)
            maps: str which is the filename of the quantitative map

        Returns:
            self.l_stats: dict , lesion statistics
            self.l_stats_df , a DataFrame consisting  of lesion stats for ilbl for all maps

        Example:
        l_stats_df.index: ilbl
        l_stats_df
            mean 	    std 	    median 	    1_quantile 	3_quantile 	min 	    max       map
         2 	0.248093 	0.004892 	0.247099 	0.245824 	0.252232 	0.239431 	0.256164  'mwf'

        """

        [self.l_stats, self.l_stats_df] = lesion_describe(self.ilbl, ilbl_mask, maps)

# session class ---------------------------------------------------------------

class Session:
    """ 
    creates a Session object
    
    """
    def __init__(self, fname: str, session_id=None):
        self.lesion_load = 0
        self.fname = fname
        self.num_lbl = 0
        self.label_img()
        self.glbl_mask = None
        self.session_id = session_id
        self.maps = {}
        self.new_ilbls= []


        if session_id is None:
            # read session_id: from file name
            if fname.index('ses-'):
                self.session_id = (self.tail.split('_')[1]).split('-')[1]
            else:
                raise ValueError(
                    "check naming of lesion masks according to bids standard example: sub-P001_ses-0m_mask.nii.gz")
        self.intersect_fu = []

    def __repr__(self):
        return str('session object'+ self.session_id)

    @property
    def glbl_intersect(self):
        if hasattr(self, '_glbl_intersect'):
            return self._glbl_intersect
        else:
            raise ValueError('No intersection calculated. Uses calc_intersect before')

    # label binary lesion masks   
    def label_img(self):
        """
        label binary image
        
        """
        # load nifty image
        self.img_file = nib.load(self.fname)
        self.mask = self.img_file.get_fdata().astype(np.int32)
        # determine lesion load of the session
        self.lesion_load = len(self.mask[self.mask == 1])
        # create session label image
        self.ilbl_mask, self.num_lbl = label(self.mask, return_num=True, connectivity=1)
        # save label image in same path from filename
        self.directory, self.tail = os.path.split(self.fname)
        self.save_ilbl_fname = join(self.directory,
                          self.tail.split('_')[0] + '_' + self.tail.split('_')[1] + f'_desc-interimlabel_mask.nii.gz')
        nib.save(nib.Nifti1Image(self.ilbl_mask.astype(np.int32), self.img_file.affine, header=self.img_file.header), self.save_ilbl_fname)
        #Creates an ilbl lesion class object, (ilbl start with 1)
        self.ilesion = [ILesion(i) for i in range(1, self.num_lbl + 1)]


    def calc_intersection(self, glbl_mask):
        """
        Function calculates the intersections of global labels with session labels 

        Parameters:
        -----------
        glbl_mask: np.array
            labeled global lesion mask


        Returns:
        --------

        glbl_intersect: list of intersecting labels of two time points [global_label1, interim_labelx]

        """
        nums = np.unique(glbl_mask)
        num_sum = len(nums[nums > 0])
        # calculate spatial intersection of label i in global label mask with label j in session label mask  
        self._glbl_intersect = intersection(glbl_mask=glbl_mask, num_sum=num_sum, label_ses_mask=self.ilbl_mask,
                                            num_ses=self.num_lbl)
        self._glbl_intersect = np.array(self._glbl_intersect)

    def _create_glesion_obj(self):
        """
        creates GlobaLesion Objects for the session

        Returns: self.glesion (list)

        """
        self.glesion = [GLesion(g, self) for g in set(self.glbl_intersect[:, 0])]

    def get_lesion(self, lesionid: int):
        """
        Returns LesionObject for given lesionid
        Args:
            lesionid: int

        Returns:
            iLesion if session has lesion with matching id
            None if no lesion could be found
        """

        for l in self.ilesion:
            if l.ilbl == lesionid:
                return l
        return None

    def calc_intersection_fu(self, fu_session):
        """
        Function calculates the intersections of session labels with follow-up session labels returns:
        self._fu_intersect: list of intersecting labels of two time points [session_interim_label, fu_interim_label]

        Parameters:
        -----------

        fu_session: str

        Example: calc_intersection_fu( '1m')

        """
        numsfu = np.unique(fu_session.ilbl_mask)
        num_ses = len(numsfu[numsfu > 0])
        # calculate spatial intersection of label i in interim label mask with label j in fu session label mask
        self._fu_intersect = intersection(glbl_mask=self.ilbl_mask, num_sum=self.num_lbl,
                                          label_ses_mask=fu_session.ilbl_mask, num_ses=num_ses)

        # determine new lesions in fu
        fu_lesion_list = numsfu[numsfu > 0]
        #[ILesion(self, i).get_new_lesions(fu_lesion_list, self._fu_intersect) for i in range(1, num_ses + 1)]
        [ILesion(i).get_new_lesions(fu_lesion_list, self._fu_intersect) for i in range(1, num_ses + 1)]

        # add child of a current lesion
        for intersect in self._fu_intersect:
            current_lesion = self.get_lesion(intersect[0])
            fu_lesion = fu_session.get_lesion(intersect[1])
            if not current_lesion is None and not fu_lesion is None:
                current_lesion.add_child(fu_lesion)

    def glbl_masks(self):
        """
        relabel lesion masks of the session with global labels, glbl_mask is saved im same directory as original
        binary segmented t2l mask

        Parameters:
        -----------
        glbl_intersect: list of intersecting labels of two time points [global_label, session_label]

        label_ses_mask: np.array
             lesion mask of the session

        Returns:
        --------
        save_fname: string 
            filname of the relabeled session mask with related global labels
        """

        # initialize mask as int, to preserve label
        self.glbl_mask = np.zeros(self.mask.shape).astype(np.int32)
        for intersect in self._glbl_intersect:
            self.glbl_mask[self.ilbl_mask == intersect[1]] = intersect[0]

        self.directory, self.tail = os.path.split(self.fname)
        self.save_glbl_mask_fname = join(self.directory,
                                         self.tail.split('_')[0] + '_' + self.tail.split('_')[1] + '_desc-glbl_mask.nii.gz')
        nib.save(nib.Nifti1Image(self.glbl_mask.astype(np.int32), self.img_file.affine, header=self.img_file.header), self.save_glbl_mask_fname)
        return self.save_glbl_mask_fname

    def add_map(self, modality: str, fname: str):
        self.maps[modality] = fname

    def add_mask(self, mask_type: str, fname: str):
        self.masks[mask_type] = fname

    def masks_dict(self):
        """
        create a dictionary masks which contains key=mask_type, value = path+fname of mask  of the session

        Returns: dict

        """

        self.masks = {'t2l': self.fname,
                      'ilbl': self.save_ilbl_fname,
                      'glbl': self.save_glbl_mask_fname}

    def found_new_lesion(self):
        """
        lesion attribute l.new is turned to True if lesion label is in session new i_lbls list

        Returns: bool

        """
        # all lesions of the session
        for l in self.ilesion:
            # list of identified new i_lbls of the session
            for i in self.new_ilbls:
                if l.ilbl==i:
                    l.new = True

    def _get_lesion_values(self):
        """
        calculates l_stats (lesion statistics) for all ilbl lesions and for a glbl of the session
        """
        # all lesions of the session self.lesion
        for l in self.ilesion:
            l.ilbl_describe( self.ilbl_mask, self.maps)
            # get all childs (ilbl) of a lesion
            l.get_all_descendents()
            # get all parents (ilbl) of a lesion
            l.get_all_ancestor()
            l.get_related_glabel(glbl_intersect=self.glbl_intersect)

        for g in self.glesion:
            g.glbl_describe( self.glbl_mask, self.maps)

# class Patient -------------------------------------------------------
class Patient:
    """ 
    creates a patient object
    
    """

    def __init__(self, subject_id: str):

        self.sessions = []
        self.subject_id = subject_id

    def get_session(self, session_id: str):
        for ses in self.sessions:
            if ses.session_id == session_id:
                return ses
        return False

    def add_session_from_file(self, fname, ses_id=None):
        ses = Session(fname, ses_id)
        self.add_session(ses)
        self.check_masks()

    def add_session(self, session):
        self.sessions.append(session)
        self.check_masks()

    def check_masks(self):

        """
        check if all masks of the patient are Nifty files and have same shape  
        """
        substring = '.nii'
        for i in range(0, len(self.sessions), 1):
            f = self.sessions[i].fname
            if f != None and substring in f:
                pass
            else:
                raise TypeError("provide .nii or nii.gz files ")
        if len(set([s.mask.shape for s in self.sessions])) != 1:
            raise ValueError(
                "time series of lesion masks have not same shapes, check lesion masks registration (same space?)")

    def create_density_mask(self):
        """
        ilbl masks (s.mask) of all sessions of the patient get summed to create a density mask. This density mask
        contains all ilbl/ new lesions of a patient with the maximum lesion load within the session.
        Density mask get binarized (bin_density) and labeled (glb_summask). glb_summask serve as reference label
        to relabel the ilbl by glbl in s.glbl_masks()

        :parameter:

        :returns: glb_summask: np.array, labeled binarized density mask --> to receive global reference lables (glbl)
                  num_glbl: int, number of glbls in patient
                  density_img: nii.gz file, sum of binarized lesion masks over time, static lesion parts
                            have higher numbers than dynamic or resolving lesion parts

        
        """

        self.density_mask = np.zeros(self.sessions[0].mask.shape)
        for s in self.sessions:
            self.density_mask = self.density_mask + s.mask
        self.density_img= nib.Nifti1Image(self.density_mask.astype(np.int32), s.img_file.affine, header=s.img_file.header)
        # binarize density mask
        self.bin_density = np.zeros(self.sessions[0].mask.shape)
        self.bin_density[self.density_mask > 0] = 1
        # create global lables from binarized density mask
        self.glb_summask, self.num_glbl = label(self.bin_density, return_num=True, connectivity=1)
        self.glb_summask_img = nib.Nifti1Image(self.glb_summask.astype(np.int32), s.img_file.affine, header=s.img_file.header)

        return self.glb_summask

    def calc_intersections(self):
        """
        calculate intersections of global label mask with session interim label mask for all sessions
        s.glbl_masks() --> relabel interim session lesion mask by corresponding glabel
        """

        self.create_density_mask()

        print('relabel lesion mask of session according to glbl intersection with session ilbls')
        waitbar = tqdm(total=len(self.sessions), desc='calculate glbl intersections')
        for s in self.sessions:
            s.calc_intersection(self.glb_summask)
            s.glbl_masks()
            waitbar.update(n=1)

    def glbl_intersect_to_df(self, path: str):
        """
        append all glbl_intersections (intersection of global lesion label with interim session label)
        for all sessions
        Returns:
        self.df_glbl_intersect: pivot table of dataframe ( rows= global labels, columns= session, value= intersecting interim
                            label or labels in case of confluent lesions)
        file: str: filename  f'sub-{subject_id}_desc-glblintersect_df.tsv  is saved in specified path containing
                            df_glbl_intersect
        """
        df_glbl = pd.DataFrame()
        for s in self.sessions:
            # create df for every session from glbl_intersect
            df = pd.DataFrame(s.glbl_intersect, columns=['glbl', 'ilbl'])
            # add session_id to df
            df['session'] = s.session_id
            # append all sessions to one df_glbl
            df_glbl = pd.concat([df_glbl, df])
        # create pivot tabel containing glbl as rows and session as columns, and corresponding interim labels as values,
        # glbl with confluent lesions contain a list of interim labels
        self.df_glbl_intersect = df_glbl.pivot_table(index='glbl', columns='session', values='ilbl', aggfunc=list)

        # sort columns of dataframe example original pivot ('0m', '1m ', '15m', '2m' ) sorted (0m, 1m, 2m ...15m)
        # sort_func needs to ensure that sessions are correctly ordered
        cols= sort_func(self.df_glbl_intersect.columns)
        self.df_glbl_intersect = self.df_glbl_intersect[cols]
        # create path folder for intersections if not exists
        if not exists(path):
            print('create path')
            os.makedirs(path)
        else:
            pass
        # save global intersections of whole patient in tsv file

        self.df_glbl_intersect.to_csv(join(path, f'sub-{self.subject_id}_desc-glblintersect_df.tsv'), sep="\t")
        print(f'saved intersections for global lesion labels and interim lesion labels of sub-{self.subject_id} '
              f'in path {path} ')
        # create a df: rows= glbl, columns = session, value= ilbl (number of confluent interim lesions in comparison to glbl )
        df_confluent = df_glbl.pivot_table(index='glbl', columns='session', values='ilbl', aggfunc='count')
        df_confluent = df_confluent[cols]
        df_confluent.to_csv(join(path, f'sub-{self.subject_id}_desc-nrconfluentinglesions_df.tsv'), sep="\t")

        return self.df_glbl_intersect, df_confluent

    def create_lbl_obj(self):
        """
        create a global lesion object and a interim label object for each session
        Returns:

        """
        for s in self.sessions:
            s._create_glesion_obj()

    def find_new_glbls(self):
        """
        identify when a g_lbl lesion first occurs in the dataframe df_glbl_intersect,
        find all new g_lbls within a session, by definition no new lesion in first session,
        returns: a dataframe of new glbl for each session, new_df.columns=['glbl', 'session']

        """
        # apply dropna row wise on df_glbl_intersect, choose first element of index = session_id
        # when lesion first appeared
        first_occur = self.df_glbl_intersect.apply(lambda x: (x.dropna().index[0]), axis=1)
        # df contain glbl as index and first occur (e.g. 0m etc.) as column
        first_occur_df = pd.DataFrame(first_occur, columns=["session"])
        for s in self.sessions:
            # list of 'new' glbl within session, (exception '0m' = pre-exisitng)
            n_glbl_list = first_occur_df[first_occur_df['session'] == s.session_id].index
            s.new_glbls= n_glbl_list

        self.new_df = pd.DataFrame()
        df = {}
        df2 = {}
        # first_occur=patients[1].first_occur
        for s in self.sessions:
            for n in list(s.new_glbls):
                # by definition no new lesion in first session
                if self.sessions[0] == s:
                    df['glbl'] = n
                    df['session'] = s.session_id
                    df['new'] = 'NaN'
                    df2 = pd.DataFrame([df])
                    self.new_df = pd.concat([self.new_df, df2])
                else:
                    df['glbl'] = n
                    df['session'] = s.session_id
                    df['new'] = 'new'
                    df2 = pd.DataFrame([df])
                    self.new_df = pd.concat([self.new_df,df2])

    def find_new_ilbls(self):
        """
        identify all i_lbls for a given list of new_glbls of the session

        Returns: list of i_lbls which are new within the session

        """
        # df value = interim label of lesion wich is new
        isnew_ilbl = self.df_glbl_intersect.apply(lambda x: x[x.dropna().index[0]], axis=1)
        isnew_ilbl_df = pd.DataFrame(isnew_ilbl, columns=['ilbl'])
        for s in self.sessions:
            ilbl_list = pd.DataFrame()
            if not s.new_glbls.any():
                # no new ilbls
                s.new_ilbls=[]
            else:
                #concat all new_ilbls of one session,
                for glbl in s.new_glbls:
                    ilbls = isnew_ilbl_df[isnew_ilbl_df['ilbl'].index == glbl]
                    ilbl_list = pd.concat([ilbl_list, ilbls])

                s.new_ilbls = ilbl_list['ilbl'].tolist()
                # flatten list of lists, list of new_ilbls per session
                s.new_ilbls = [label for labels in s.new_ilbls for label in labels]


    def new_ilbl(self):
        """
        run session function found_new_lesions for all sessions
        Returns:

        """
        for s in self.sessions:
            if not s.session_id == str(self.sessions[0]):
                s.found_new_lesion()

    def find_vanishing_glbls(self):
        """
        identify when a g_lbl lesion last occurs in the dataframe df_glbl_intersect,
        find all vanishing g_lbls within a session, returns: a dataframe in which lesion last occured
        by definition, a lesion can't be vanished in last session
        self.vanished_df.columns = ['glbl' , 'session']
        """

        last_occur = self.df_glbl_intersect.apply(lambda x: (x.dropna().index[-1]), axis=1)
        last_occur_df = pd.DataFrame(last_occur, columns=["session"])
        for s in self.sessions:

            v_glbl_list = last_occur_df[last_occur_df['session'] == s.session_id].index
            s.vanished_glbls= v_glbl_list

        self.vanished_df = pd.DataFrame()
        df = {}
        for s in self.sessions:
            for n in list(s.vanished_glbls):
                if self.sessions[-1] == s:
                    df['glbl'] = n
                    df['session'] = s.session_id
                    df['vanished'] = 'NaN'
                    df2 = pd.DataFrame([df])
                    self.vanished_df = pd.concat([self.vanished_df, df2])
                else:
                    df['glbl'] = n
                    df['session'] = s.session_id
                    df['vanished'] = 'vanished'
                    df2 = pd.DataFrame([df])
                    self.vanished_df = pd.concat([self.vanished_df, df2])

    def calc_intersections_fu(self):
        """
        calculate intersection of interim label with session follow-up label
        (last time-point has by definition no intersections with follow-up labels)

        """
        print('calculate intersection of interim labels with session follow-up  interim labels')
        waitbar = tqdm(total=len(self.sessions), desc='calculate fu intersections')
        i = 0
        for s in self.sessions:
            i = i + 1
            if s == self.sessions[-1]:
                # last session has by definition no fu_intersect
                self.sessions[i - 1]._fu_intersect = []
            else:
                s.calc_intersection_fu(self.sessions[i])
            waitbar.update(n=1)

    def get_lesion_values(self):
        """
        calculates lesion metrics for (ilbl and glbl) for all sessions of the patient and save it in lesion
        or glesion object
        Returns:

        example:
        patient.get_lesion_values()
        # df with ilbl stats on lesion class
        patient.sessions[0].ilesion[0].l_stats_df
        # glbl stats for glesion
        patient.sessions[0].glesion[0].l_stats_df
        """
        waitbar = tqdm(total=len(self.sessions), desc='get lesion values')
        for s in self.sessions:
            s._get_lesion_values()
            waitbar.update(n=1)

    def long_df_lesion_values(self):
        """
        collect df of individual glbl and ilbl from glesion  or ilesion object.


        Returns:
        glbl_stats_df: pd.DataFrame, consists mean, std, median, 1_quartile, 2_quartile,
                                               3_quartile, 4_quartile, min max for all glbls and all sessions
        ilbl_stats_df: pd.DataFrame, consists mean, std, median, 1_quartile, 2_quartile,
                                               3_quartile, 4_quartile, min max for all ilbls and all sessions

        """
        self.glbl_stats_df = pd.DataFrame()
        self.ilbl_stats_df = pd.DataFrame()
        childs=[]
        parents=[]
        glbl=[]


        for s in self.sessions:
            glbls = len(set(s.glbl_intersect[:, 0]))
            for g in range(0, glbls):
                # add session_id to df
                s.glesion[g].l_stats_df['session'] = s.session_id
                self.glbl_stats_df = pd.concat([self.glbl_stats_df, s.glesion[g].l_stats_df])
            ilbls = len(set(s.glbl_intersect[:, 1]))
            for i in range(0, ilbls):
                # add session_id to df
                s.ilesion[i].l_stats_df['session'] = s.session_id
                # add child ilbl label (descendent lesion) and convert list to string
                childs= s.ilesion[i].childs
                childstr = ", ".join(repr(e) for e in childs)
                s.ilesion[i].l_stats_df['child_ilbl'] = childstr
                # add parent ilbl label (ancestor lesion) and convert list to string
                parents = s.ilesion[i].parents
                parentstr = ", ".join(repr(e) for e in parents)
                s.ilesion[i].l_stats_df['parent_ilbl'] = parentstr
                # add corresponding glbl
                glbl= s.ilesion[i].glbl
                s.ilesion[i].l_stats_df['corresp_glbl'] = glbl
                self.ilbl_stats_df = pd.concat([self.ilbl_stats_df, s.ilesion[i].l_stats_df])

        self.glbl_stats_df.index.name = 'glbl'

        # make 'glbl' to a column in df
        self.glbl_stats_df= self.glbl_stats_df.reset_index()
        merged = self.glbl_stats_df.merge(self.new_df, how='left', left_on=['glbl', 'session'], right_on=['glbl', 'session'])
        merged2 = merged.merge(self.vanished_df, how='left', left_on=['glbl', 'session'], right_on=['glbl', 'session'])
        self.glbl_stats_df= merged2

        self.ilbl_stats_df.index.name = 'ilbl'
        # make 'ilbl' to a column in df
        self.ilbl_stats_df= self.ilbl_stats_df.reset_index()

        self.glbl_stats_df['month'] = pd.to_numeric(self.glbl_stats_df['session'].str.replace('m', '').astype('int'))
        self.ilbl_stats_df['month'] = pd.to_numeric(self.ilbl_stats_df['session'].str.replace('m', '').astype('int'))
        # order columns of df



    def create_4D_images(self):

        """
        This function produce a dict containing lists of session images of maptype (map types and mask types)
        which should be concatenated to a 4d image.
        The maps, masks must to be specified via the config_file.json
        4D files get concatenated and saved in function save_4d_images()

        Parameter:


        Returns:
            self.create_4D_img: dict (key = maptype, value = list of images from time serie)

        Example:

        create_4D_image= {'mwf': ['/home/jovyan/work/AFIL/derivatives/brainex_despot/sub-DEV001/ses-0m/sub-DEV001_ses-0m_space-pattmp_mwf.nii.gz',
                                  '/home/jovyan/work/AFIL/derivatives/brainex_despot/sub-DEV001/ses-1m/sub-DEV001_ses-1m_space-pattmp_mwf.nii.gz',
                                  '/home/jovyan/work/AFIL/derivatives/brainex_despot/sub-DEV001/ses-2m/sub-DEV001_ses-2m_space-pattmp_mwf.nii.gz',]
                         't2lmask': ['/home/jovyan/work/AFIL/derivatives/segmentation/sub-DEV001/ses-0m/sub-DEV001_ses-0m_space-pattmp_t2lmask.nii.gz',
                                  '/home/jovyan/work/AFIL/derivatives/segmentation/sub-DEV001/ses-1m/sub-DEV001_ses-1m_space-pattmp_t2lmask.nii.gz',
                                  '/home/jovyan/work/AFIL/derivatives/segmentation/sub-DEV001/ses-2m/sub-DEV001_ses-2m_space-pattmp_t2lmask.nii.gz',]

        """
        self.create_4D_image = {}
        for s in self.sessions:
            # masks_dict() create a dict s.masks containing, t2lmask, ilbl, glbl mask per session
            s.masks_dict()
            #create_4D_image = {}
            for images in [s.maps, s.masks]:
                for maptype in images.keys():
                    # create empty dict with maptypes
                    if not maptype in self.create_4D_image.keys():
                        self.create_4D_image[maptype] = []
                    # add image path of session to list for every maptype
                    self.create_4D_image[maptype].append(images[maptype])

class GLesion():
    """ create a global lesion object of the time series
    """

    def __init__(self, glbl, session):
        self.glbl = glbl
        self.session = session
        self.new_glabel = False
        self.l_stats_df = None

    def __repr__(self):
        return str(self.glbl)

    def glbl_describe(self, mask, maps):

        """
            This function masks the lesion area in the quantitative map and calculates
            some lesion statistics based on provided stat_func().

            Args:
                self.glbl: int global label lesion id
                mask: numpy.ndarray labeled lesion mask (glbl)
                maps: dict map_key=modality, value= str, filename of the quantitative map
            Returns:  self.l_stats: dict , lesion statistics
                      l_stats_df , a DataFrame consisting of one row index == label, and lesion stats
            Example:
                l_stats_df
             	mean 	    std 	    median 	    1_quantile 	3_quantile 	min 	    max       map
            2 	0.248093 	0.004892 	0.247099 	0.245824 	0.252232 	0.239431 	0.256164  'mwf'

      """
        [self.l_stats, self.l_stats_df] = lesion_describe(self.glbl, mask, maps)


# class project -------------------------------------------------------
class Project:
    """
    creates a project object

    """
    def __init__(self, config_file: str):
        self.config = config_file
        self.projectdir = None
        self.load_config_from_json()
        self.get_config_values()
        self.patients = dict()
        self.sort= sort_func

    def load_config_from_json(self):
        """
        load projectdir, subject_list and session list from config.json file

        Find for all maps for all sessions, described by json entries.

        Args:


        Returns:
            projectdir

        Examples
        project=Project(config_file="/code/AFIL_config.json")
        ses_list= project.sessions

        Example for json:
        {   "projectdir": "/home/jovyan/work/AFIL",
            "subjects": "DEV008",
            "sessions": [
                    "0m",
                    "1m",
                    "2m",
                    "3m",
                    "4m",
                    "5m",
                    "6m",
                    ],
            "masks": {
                "t2l": "/derivatives/segmentation/sub-{subject}/ses-{session}/anat/sub-{subject}_ses-{session}_space-pattmp_t2lmask.nii.gz"
                },
            "maps": {
                "mwf": "/derivatives/brainex_despot/sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_space-pattmp_mwf.nii.gz"
                }
        }
        """
        with open(self.config, "r") as file:
            import json
            self.load_config = json.load(file, strict=False)
            if self.projectdir is None:
                self.projectdir = self.load_config['projectdir']

    def get_config_values(self):

        self.subjects = self.load_config['subjects']
        self.sessions = self.load_config['sessions']
        self.load_masks = self.load_config['load_masks']
        self.load_maps = self.load_config['load_maps']
        self.fdirs = self.load_config['fdirs']

    def create_sessions(self, subject_id: str, force= False):
        """
        create patient object containing all session t2 lesion masks

        Args: subject_id : str

        Returns:

        Examples
        project=Project(config_file=projectdir+"/code/AFIL_config.json")
        project.create_sessions(subject_id='DEV008')
        """
        patient = Patient(subject_id=subject_id)
        # sessions get sorted
        sessions = self.sort(self.sessions)

        for s in sessions:
            file = self.projectdir + self.load_masks['t2l'].format(subject=patient.subject_id, session=s)
            patient.add_session_from_file(file, s)

        # add modality maps from file and add them to sessions
        self.load_all_maps(patient)
        # process afil
        self.process_afil(patient, force)
        return patient

    def load_all_maps(self, patient: Patient):
        """
        Find for all maps for all sessions, described by dictonary entries in .json config file.
        Use {session} and {subject} to replace the parts of the filenames with current values
        subject_id is specified in create_sessions(self, subject_id), load_maps: dict of str
            key=name of the maps, value=search_string for filename

        Args:
            patient: Patient, patient object for which maps are loaded

        Returns:


        Examples
        load_maps = {'mwf': '/derivatives/brainex_despot/sub-{subject}/ses-{session}m/sub-{subject}_ses-{session}m_space-pattmp_mwf.nii.gz'}
        project.load_all_maps()
        project.patient.sessions[0].maps['mwf']= /home/jovyan/work/AFIL/derivatives/brainex_despot/sub-DEV008/ses-0m/sub-DEV008_ses-0m_space-pattmp_mwf.nii.gz
        """

        for s in range(0, len(patient.sessions)):
            for key in self.load_maps:
                fname = self.load_maps[key].format(subject=patient.subject_id, session=patient.sessions[s].session_id)
                # full path and filename of map
                fname = self.projectdir + fname
                fname = fname
                if exists(fname):
                    patient.sessions[s].add_map(key, fname)

    def process_afil(self, patient: Patient, force= False):
        """
        This function starts main afil functions to process lesion tracking
        Args: force: bool , if mode force (overwrite), re-run afil processing and ignore already processed files
        Returns:

        """
        # check if subject has already been processed
        path= join(self.projectdir, self.fdirs['export_lesion_stats'])
        fname1 = f'sub-{patient.subject_id}_desc-longitudinal_label-glbl_lesionstats.xlsx'
        fname2 = f'sub-{patient.subject_id}_desc-longitudinal_label-ilbl_lesionstats.xlsx'
        fpath1 = join(path, fname1)
        fpath2 = join(path, fname2)
        # if export files exists, pass
        if isfile(fpath1) & isfile(fpath2) & (not force) :
            print(f'subject {patient.subject_id} is already processed')
            print(f'found files sub-{patient.subject_id}_desc-longitudinal_label-glbl_lesionstats.xlsx')
            print(f'found files sub-{patient.subject_id}_desc-longitudinal_label-ilbl_lesionstats.xlsx')
        else:
            #try:
            print(f'run_afil: lesion tracking on subject {patient.subject_id}')
            self.save_load_glbl_intersect_to_df(patient, force)
            self.save_load_fu_intersect(patient, force)
            self.save_densitymask(patient)
            self.save_load_glbl_masks(patient)
            self.save_4d_images(patient)

            # find new glbl of patient
            patient.find_new_glbls()
            # find new ilbl of session
            patient.find_new_ilbls()
            patient.new_ilbl()
            patient.find_vanishing_glbls()

            # create glbl lesion objects
            patient.create_lbl_obj()

            # calc ilbl & glbl metrics for whole patient
            patient.get_lesion_values()
            self.export_lesion_stats(patient)
            #except:
            #    print('check if input data (lesion masks, and quantitative maps are available )')
            #    print('check naming of config file paths ')
            #    print('for bids datasets check if dataset discription.json is in derivatives folders')

    def save_densitymask(self, patient: Patient):
        """
        save the lesion density mask which is a result of the addition of all binary lesion masks over
        time. Label one represents lesion areas which are present only in one time point and may disappear.
        Labels (number) >1 areas represents lesion areas which are present at multiple (= number) time-points.
        path: str, path folder where to save the density image, self.save_density_fname: str (filename )
                 self.save_glb_summask_fname : str (filename )

        Args:


        Returns: patient

        Example:

        patient.save_density_mask()


        """

        path = join(self.projectdir, self.fdirs['density_mask'], f'sub-{patient.subject_id}')

        if not exists(path):
            os.makedirs(path)
        self.save_density_fname = join(path, f'sub-{patient.subject_id}_desc-density_mask.nii.gz')
        if isfile(self.save_density_fname):
            pass
        else:
            patient.create_density_mask()
            nib.save(patient.density_img, self.save_density_fname)
        self.save_glb_summask_fname = join(path, f'sub-{patient.subject_id}_desc-glbreference_mask.nii.gz')
        if isfile(self.save_glb_summask_fname):
            pass
        else:
            patient.create_density_mask()
            nib.save(patient.glb_summask_img, self.save_glb_summask_fname)
        return patient

    def save_load_glbl_intersect_to_df(self, patient: Patient, force= False):
        """
        function for loading and saving corresponding lesion label intersections from ilbl with glbl for all
        sessions of a patient
        if sub-x_desc-glblintersect_df.tsv file is available, than glbl_intersect get loaded, otherwise
        intersection of all sessions get processed and saved

        Args: patient: Patient class object
              force: bool, default False, for overwriting set to True from commandline program -f

        Returns: patient.sessions[count]._glbl_intersect : list

        Example: project.save_load_glbl_intersect_to_df()
        """

        path = join(self.projectdir, self.fdirs['save_load_glbl_intersect'], f'sub-{patient.subject_id}')
        fname = f'sub-{patient.subject_id}_desc-glblintersect_df.tsv'
        fpath = join(path, fname)

        if isfile(fpath) & (not force):
            # load glbl_intersect_to_df into patient object
            df = pd.DataFrame(pd.read_csv(join(fpath), sep='\t', index_col='glbl'))
            ## caution when reading dataframes! --> int --> list of strings
            ## applymap is a  function operating cell wise
            ## literal_eval remove quotation marks from str [['2,4'], ['3']] --> [[2,4],[3]]
            df = df.applymap(lambda x: ast.literal_eval(x), na_action='ignore')
            patient.df_glbl_intersect = df
            df2 = df.reset_index()
            for count, s in enumerate(patient.sessions):
                ses= s.session_id
                # filter intersections for every session
                df_col = df2[['glbl',ses]].dropna()
                ## unpivot by df.explode Transform each element of a list-like (multiple intersections) to a row
                patient.sessions[count]._glbl_intersect = np.array(df_col.explode(ses))
        else:
            # if glbl_intersect_to_df does not exist, than create makedirs and calc intersections
            if not exists(path):
                os.makedirs(path)
            # calculates global label (glbl) intersection with intrim label (ilbl) for every session
            patient.calc_intersections()
            patient.glbl_intersect_to_df(path)
        return patient

    def save_load_fu_intersect(self, patient: Patient, force= False):
        """
        all ilbl intersections with follow-up ilbls of all sessions of the patient get collected
        in on .json file
        Returns:
            patient.sessions[count]._fu_intersect: list

        Example: project.save_load_fu_intersect()
        """

        path = join(self.projectdir, self.fdirs['save_load_fu_intersect'], f'sub-{patient.subject_id}')
        fname = f'sub-{patient.subject_id}_desc-fuintersect.json'
        fpath = join(path, fname)
        # if fu_intersect.json exists
        if isfile(fpath) & (not force):
            with open(fpath) as json_file:
                data= json.load(json_file)
            # load _fu_intersect from json into sessions
            for count, s in enumerate(patient.sessions):
                ses = s.session_id
                try:
                    patient.sessions[count]._fu_intersect = data[ses]
                except:
                    print('delete invalid fuintersect.json')
        else:
            # if sub-x_desc-fuintersect.json doesn't exists
            if not exists(path):
                os.makedirs(path)
            # calc follow_up intersections
            patient.calc_intersections_fu()
            #save intersections of all sessions in one json file
            collect_fu_intersect={"session_id":[], "fu_intersect": []}
            for count, s in enumerate(patient.sessions):
                ses = s.session_id
                # reduce array for json.dump
                #conv = self.patient(p).sessions[count]._fu_intersect
                #collect_fu_intersect[ses]= [list(i) for i in conv ]
                collect_fu_intersect[ses]= patient.sessions[count]._fu_intersect
            # save json file
            with open(fpath, 'w') as fp:
                json.dump(collect_fu_intersect, fp)
        return patient

    def save_load_glbl_masks(self, patient: Patient):
        """
        globally labeled lesion masks (glbl masks) get loaded for all sessions of the patient
        or get processed if not present for the patient, get path of glbl_masks via config file self.fdirs['save_load_glblmasks']

        Returns:
            self.patient(p).sessions[s].glbl_mask
            self.patient(p).sessions[s].save_glbl_mask_fname

        Example:
            project.save_load_glbl_masks()
        """


        for count, s in enumerate(patient.sessions):
            path = join(self.projectdir, self.fdirs['save_load_glblmasks'], f'sub-{patient.subject_id}',
                        f'ses-{s.session_id}', 'anat')
            fname = f'sub-{patient.subject_id}_ses-{s.session_id}_desc-glbl_mask.nii.gz'
            fpath = join(path, fname)
            if isfile(fpath):
                #print('found glbl_masks', fpath)
                try:
                    patient.sessions[count].glbl_mask = nib.load(fpath).get_fdata()
                    patient.sessions[count].save_glbl_mask_fname = fpath
                except:
                    print('check if valid glbl masks exists in specified path for all sessions, otherwise delete them all for the patient')
            else:
                # calc glbl masks for all sessions
                patient.sessions[count].glbl_masks()
        return patient

    def save_4d_images(self, patient: Patient):
        """
            concatenate session images for every multimodal map and mask to a 4d nii.gz image and save it in specified path
            The maps, masks and path for saving the 4D images must be specified via the config_file.json
            (via config.json file @--> "fdirs": {
                                               "4dimages": "derivatives/4dimage",
                                               ...
                                              }
                Returns: patient object

                example:
                project.save_4d_images()
                """

        path = join(self.projectdir, self.fdirs['4dimages'], f'sub-{patient.subject_id}')
        if not exists(path):
            os.makedirs(path)
        patient.create_4D_images()
        try:
            for type in patient.create_4D_image:
                # create list of image paths for every maptype
                lst = patient.create_4D_image[type]
                # create 4D nifty image
                concat = nib.funcs.concat_images(lst)
                nib.save(concat, join(path, f'sub-{patient.subject_id}_desc-{type}_4D.nii.gz'))
        except Exception as e:
            # ... PRINT THE ERROR MESSAGE ... #
            print(f'not processed sub-{patient.subject_id}_desc-{type}_4D.nii.gz because of {e}')
        return patient

    def export_lesion_stats(self, patient: Patient ):
        """
        save patient.glbl_stats_df and patient.ilbl_stats_df in individual excel files
        dataframes containing whole longitudinal lesion metrics (glbl or ilbl) collected from l_stats_df,
        define path for saving lesion stats via config_file.json path. Two excel files at specified folder location
        sub-xxx_desc-longitudinal_label-glbl_lesionstats.xlsx and sub-xxx_desc-longitudinal_label-ilbl_lesionstats.xlsx
        get saved

        Args: Patient class object

        example:
        project.export_lesion_stats(patient))
        """
        path = join(self.projectdir, self.fdirs['export_lesion_stats'])
        patient.long_df_lesion_values()
        # order columns of patient.glbl_stats_df & patient.ilbl_stats_df
        patient.glbl_stats_df= patient.glbl_stats_df[['glbl','volume','map','mean', 'std', 'median', '1_quantile',
                                                '3_quantile', 'min', 'max', 'new', 'vanished', 'session', 'month']]
        patient.ilbl_stats_df= patient.ilbl_stats_df[['ilbl','volume','map','mean', 'std', 'median', '1_quantile',
                                                '3_quantile', 'min', 'max', 'session', 'month','child_ilbl',
                                                'parent_ilbl', 'corresp_glbl']]
        # export df
        if not exists(join(path)):
            os.makedirs(path)
        for i in [patient.glbl_stats_df, patient.ilbl_stats_df]:
            df_columns= i.columns.values.tolist()
            label= df_columns[0]
            fname = f'sub-{patient.subject_id}_desc-longitudinal_label-{label}_lesionstats.xlsx'
            i.to_excel(join(path, fname), na_rep='NaN', index=False)
            print('saved lesionstats:', join(path,fname))

class BidsProject(Project):

    """
    create a BIDS Project

    Example for .json config file:
    {
        "projectdir": "/home/jovyan/work/AFIL",
        "ignore_paths": ["code", "tmp"],
        "bidslayout" : {
            "derivatives": true
            },
        "load_masks": {
            "t2l": {
                "suffix": "t2lmask",
                "extension": "nii.gz"
                }
            },
        "load_maps": {
            "mwf": {
                "suffix":"mwf",
                "extension": "nii.gz"
                },
            "qT1": {
                "suffix":"qT1",
                "extension": "nii.gz"
                }
            },
        "load_save_fdir": {
            "4dimages": "derivatives/4dimage",
            "export_lesion_stats": "derivatives/lesion_stats",
            "density_mask": "derivatives/density_mask",
            "save_load_glbl_intersect": "derivatives/glbl_intersect",
            "save_load_glblmasks": "derivatives/segmentation",
            "save_load_fu_intersect": "derivatives/fu_intersect"
            }
    }
    """

    def get_config_values(self):
        import bids
        #bids.config.set_option('extension_initial_dot', True)
        from bids.layout.index import BIDSLayoutIndexer
        # from bids.layout import BIDSLayout
        """
        get values via BIDS from config .json file, (subjects, sessions, masks, maps)

        """
        self.ignore_paths = ["tmp"]
        if "ignore_paths" in self.load_config.keys():
            self.ignore_paths = self.load_config["ignore_paths"]
        self.bidslayout = self.load_config['bidslayout']

        self.layout = bids.layout.BIDSLayout(root=self.projectdir, **self.bidslayout,
                                 indexer=BIDSLayoutIndexer(validate=False,
                                                           index_metadata=False,
                                                           ignore=self.ignore_paths))
        print(self.layout.get(return_type='filename'))
        # get all subjects and sessions

        self.subjects = self.layout.get_subjects()
        self.sessions = self.layout.get_sessions()
        # load config for maks and maps from .json
        # multiple mask & maps types are allowed
        self.load_masks = self.load_config['load_masks']
        self.load_maps = self.load_config['load_maps']

        # load all maps and masks specified in .json file
        # remind this will find all maps from all patients and sessions of the project

        self.masks = {}
        for key in self.load_masks:
            self.masks[key] = self.layout.get(**self.load_masks[key], return_type='file')
        self.maps = {}
        for key in self.load_maps:
            self.maps[key] = self.layout.get(**self.load_maps[key], return_type='file')

        self.fdirs = self.load_config['fdirs']


    def create_sessions(self, subject_id: str, force= False):
        """
                create patient object containing all session t2 lesion masks
                Args: subject_id : str
                      force: bool , if mode force (overwrite), re-run afil processing and ignore already processed files

                Returns:

                Examples
                project=BidsProject(config_file=projectdir+"/code/AFIL_config_testbids.json")
                project.create_sessions(subject_id='DEV008')
        """

        patient = Patient(subject_id=subject_id)
        sessions=self.layout.get_sessions(subject=subject_id)
        sessions=self.sort(sessions)

        for ses in sessions:
            file = self.layout.get(subject=subject_id, session=ses, suffix=self.load_masks['t2l']['suffix'],
                                   extension='nii.gz', return_type='file')
            file = file[0]
            patient.add_session_from_file(file, ses)

        # add modality file maps from specific subject_id and add them to sessions
        self.load_all_maps(patient)
        self.process_afil(patient, force)
        return patient

    def load_all_maps(self, patient):
        """
        load maps specified in .json file (load_maps) for a specific subject_id (specified in
        create_sessions(self, subject_id) and add them to the session

        Returns:
        patient.sessions[0].maps[key]: str ( filepath of the map)

        Example
        project.load_all_maps()
        project.patient.sessions[0].maps['mwf']= '/home/jovyan/work/AFIL/derivatives/brainex_despot/sub-DEV008/ses-0m/sub-DEV008_ses-0m_space-pattmp_mwf.nii.gz'
        """
        for s in range(0, len(patient.sessions)):
            for key in self.load_maps:
                fname = self.layout.get(**self.load_maps[key], subject=patient.subject_id,
                                        session=patient.sessions[s].session_id, return_type='file')

                fname = fname[0]
                if exists(fname):
                    patient.sessions[s].add_map(key, fname)
