# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:59:27 2019
     This script process lesion tracking afil for subjects in bids data structure or spezified subject & session list for no bids version
     maps and masks have to be specified in the config.json file
    
    USAGE:
    # bids
        python3 run_afil.py -c /code/afil/AFIL_config_bids.json --> will run all subjects found by bids
        python3 run_afil.py -c /code/afil/AFIL_config_bids.json -s DEV034 -f --> re-run a specific subject
    # nobids
        python3 run_afil.py -c /code/afil/AFIL_config.json -n 
    
@author: koehlerca
"""


def afil(subjects, project, force=False):
    import nibabel as nib
    from skimage.measure import label
    from skimage import io
    import json
    import pandas as pd
    import numpy as np
    from nilearn.image import image
    import scipy as sp

    patients=[]
    
    for sub in subjects:
        print(sub)
        
        try:
            patient = project.create_sessions(subject_id=sub, force= force)
            patients.append(patient)
        except Exception  as e:
            print(f' ###! AFIL WENT WRONG FOR SUBJECT {sub} !### because of', e)
            print('# a common error is the incomplete number of files for all sessions, segmentation masks and maps \n')
            print('# if one maptye is incomplete delete the specified map type in config file or delete the session with missing data in segmentation and maps folder\n ')
            print('# file naming has to be consistent in bids format otherwise the bids layout grabber wont find the files ')    

def main():
    import argparse
    import os
    import ipdb 
    from os.path import abspath, dirname, join, isfile, exists
    from afil.AFIL_class import intersection, ILesion, GLesion, Session, Patient, Project, BidsProject, sort_func
    


    parser = argparse.ArgumentParser(description="run lesion tracking afil")
    parser.add_argument('-s' ,'--subject', type= str, nargs='*', default=None, help='subjects DEV001')
    parser.add_argument('-f' ,'--force', action='store_true', default= False, help= 're-run whole afil processing of a subject overwrite lesion_stats')
    parser.add_argument('-c' ,'--config', type= str, nargs= 1, default='AFIL_config_bids.json', help='AFIL_config_bids.json')
    #parser.add_argument('-p' ,'--path', type= str, nargs= 1, help='provide projectdir')
    parser.add_argument('-n' ,'--nobids', action='store_true', help='input data is in bids data structure ')
    
    args = parser.parse_args()
    projectdir= dirname(dirname(abspath('__file__')))
    print('projectdir', projectdir)
    print(args.config[0])
    
    config= args.config[0]
    subjects = args.subject
    bids= not args.nobids 
    # load config_file, 
    
    if bids:
        project=BidsProject(config_file=projectdir+config)
    else:    
        project=Project(config_file=config)
        
    if subjects is None:
        subjects = project.subjects
        

    afil(subjects, project, args.force)


if __name__ == '__main__':
    main()
