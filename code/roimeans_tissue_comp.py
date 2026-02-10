# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:59:27 2019

@author: koehlerca
"""


def run_wf(layout, subjects):
    import os
    from bids.layout import BIDSLayout
    from os.path import abspath, dirname, join
    from nipype import Node, DataGrabber, Workflow, Function, MapNode, DataSink, Rename
    from nipype.interfaces.io import  BIDSDataGrabber
    from nipype.interfaces.utility import IdentityInterface, Merge
    from nipype.interfaces.ants import N4BiasFieldCorrection
    from nipype.interfaces.fsl import MultiImageMaths, Threshold
    from wmi_nipype_workflows.image_statistik import get_roi_means
    from wmi_nipype_workflows.preproc import dawm_wf, remove_small_dawm, calc_dawm_thresh
    from wmi_nipype_workflows.brainextract_workflow import brain_extraction_wf
    
    projectdir = dirname(dirname(abspath(__file__)))

    #layout = BIDSLayout(projectdir, derivatives=True, ignore=['code', 'tmp'], validate=False)
    # specify sessions of a subject
    sessions = layout.get_sessions(subject= subjects)
    
    #sessions = ['0m', '2m', '3m', '6m', '15m', '21m', '27m', '33m', '39m', '51m', '63m', '75m']
    #sessions = ['63m']
    #subjects = ['DEV001']
    #subjects = layout.get_subjects()



    infosource = Node(IdentityInterface(fields=['subject_id', 'session_id']),
                          name='infosource')
    infosource.iterables = [('subject_id', subjects),
                            ('session_id', sessions)]

    load_files = Node(DataGrabber(infields=['subject_id', 'session_id'],
                            outfields=['t1w','flair', 'pd','t1c', 't2l', 'wm','mwf']),
                            name='load_files')
                                   
    load_files.inputs.template = '*'
    load_files.inputs.sort_filelist = True
    load_files.inputs.template_args = dict(t1w = [['subject_id', 'session_id', 'subject_id', 'session_id']],
                                               flair = [['subject_id', 'session_id', 'subject_id', 'session_id']],
                                               pd = [['subject_id', 'session_id', 'subject_id', 'session_id']],
                                               t1c= [['subject_id', 'session_id', 'subject_id', 'session_id']],
                                               mwf = [['subject_id', 'session_id', 'subject_id', 'session_id']],
                                               t2l = [['subject_id', 'session_id', 'subject_id', 'session_id']],
                                               wm = [['subject_id', 'session_id', 'subject_id', 'session_id']],
                                               )
        
    load_files.inputs.field_template = dict(t1w= projectdir+ '/derivatives/preprocessing/sub-%s/ses-%s/anat/sub-%s_ses-%s_space-pattmp_T1w.nii.gz',
                                                flair = projectdir+ '/derivatives/preprocessing/sub-%s/ses-%s/anat/sub-%s_ses-%s_space-pattmp_FLAIR.nii.gz',
                                                pd = projectdir+ '/derivatives/preprocessing/sub-%s/ses-%s/anat/sub-%s_ses-%s_space-pattmp_PD.nii.gz',
                                                t1c = projectdir+ '/derivatives/preprocessing/sub-%s/ses-%s/anat/sub-%s_ses-%s_space-pattmp_ce-gd_T1w.nii.gz',
                                                mwf = projectdir + '/derivatives/mcdespot/sub-%s/ses-%s/anat/sub-%s_ses-%s_space-pattmp_mwf.nii.gz',
                                                t2l = projectdir+ '/derivatives/segmentation/sub-%s/ses-%s/anat/sub-%s_ses-%s_space-pattmp_t2lmask.nii.gz',
                                                wm = projectdir+ '/derivatives/segmentation/sub-%s/ses-%s/anat/sub-%s_ses-%s_space-pattmp_wmmask.nii.gz'
                                                )

    #wf.write_graph()

    #wf.run()
    
    ##NODES
    from nipype.interfaces.fsl.maths import ApplyMask
    brain_ex= brain_extraction_wf()

    applymask_flair = Node(ApplyMask(), name='applymask_flair')
    applymask_t1w = Node(ApplyMask(), name='applymask_t1w')
    applymask_pd = Node(ApplyMask(), name='applymask_pd')
    applymask_t1c = Node(ApplyMask(), name='applymask_t1c')
    applymask_mwf = Node(ApplyMask(), name='applymask_mwf')


    dawm = dawm_wf()
    merge_list = Node(Merge(2), name= 'merge_list')
    nawm = Node(MultiImageMaths(), name='nawm')
    nawm.inputs.op_string = " -sub %s -sub %s"

    # prevent negative values in nawm mask du to -sub
    nawm_bin = Node(Threshold(), name='nawm_bin')
    nawm_bin.inputs.thresh = 0

    # read out lesion values/  get_roi_means is a function not a workflow
    roi_means = Function(function=get_roi_means, input_names=['mapfile', 'roifile'], output_names=['output', 'stats_array'])
    roimeans_t2l = Node(roi_means, name='roimeans_t2l')
    roimeans_nawm = Node(roi_means, name='roimeans_nawm')
    roimeans_dawm = Node(roi_means, name='roimeans_dawm')
    
    
    ## WORKFLOWS
    wf = Workflow(name='Workflow', base_dir=join(projectdir, 'tmp'))
    wf.connect([(infosource, load_files, [('subject_id','subject_id'),
                                          ('session_id', 'session_id')]),  
                (load_files, brain_ex, [('t1w', 'inputnode.in_file')]),
                (load_files, applymask_t1w, [('t1w', 'in_file')]),
                (brain_ex, applymask_t1w, [('outputnode.mask', 'mask_file')]),
                (load_files, applymask_flair, [('flair', 'in_file')]),
                (brain_ex, applymask_flair, [('outputnode.mask', 'mask_file')]),
                (load_files, applymask_pd, [('pd', 'in_file')]),
                (brain_ex, applymask_pd, [('outputnode.mask', 'mask_file')]),
                (load_files, applymask_t1c, [('t1c', 'in_file')]),
                (brain_ex, applymask_t1c, [('outputnode.mask', 'mask_file')]),
                (load_files, applymask_mwf, [('mwf', 'in_file')]),       
                (brain_ex, applymask_mwf, [('outputnode.mask', 'mask_file')]),
                
                
                
                (applymask_flair, dawm, [('out_file','inputnode.in_file')]),
                (load_files, dawm, [('t2l','inputnode.t2l'),
                                    ('wm', 'inputnode.wm')]), 
                (load_files, merge_list, [('t2l','in1')]),
                #(dawm, merge_list, [('rmsmdawm.bin_file', 'in2')]),
                (dawm, merge_list, [('outputnode.out_file', 'in2')]),  
                (merge_list, nawm, [("out", "operand_files")]),
                (load_files, nawm, [("wm", "in_file")]),
                (nawm, nawm_bin, [("out_file", "in_file")]),
                (nawm_bin, roimeans_nawm, [("out_file", "roifile")]),
                (load_files, roimeans_nawm, [('mwf', 'mapfile')]),
                (load_files, roimeans_t2l, [('t2l', 'roifile')]),
                (load_files, roimeans_t2l, [('mwf', 'mapfile')]),
                (dawm, roimeans_dawm, [('rmsmdawm.bin_file', 'roifile')]),
                (load_files, roimeans_dawm, [('mwf', 'mapfile')]),
                
                
               ])


    def container_name(subject_id, session_id=None):
        """ Generate container-name for datasink. If session_id is not given no
        ses- subfolder wil be used.


        Parameters
        ----------
        subject_id: str
            Plain SubjectID (without sub-)
        session_id: str (optional)
            Give SessionID (without ses-).

        Return
        ------

        str

        """
        from os.path import join

        container = "sub-%s" %(subject_id)
        if not session_id is None:
            container = join(container, "ses-%s" %(session_id))
            return container
        
        return container

    
    
    # layout = BIDSLayout(projectdir, derivatives=True, ignore=['code'], validate=False)

    segmentation_sink = Node(DataSink(base_directory=join(projectdir,'derivatives', 'segmentation'),
                                    remove_dest_dir=True,
                                    parameterization=False),
                            name='segmentation_sink') 

    brainex_sink = Node(DataSink(base_directory=join(projectdir,'derivatives', 'brain_extraction'),
                                    remove_dest_dir=True,
                                    parameterization=False),
                            name='brainex_sink') 

    brainexdespot_sink = Node(DataSink(base_directory=join(projectdir,'derivatives', 'brainex_despot'),
                                    remove_dest_dir=True,
                                    parameterization=False),
                            name='brainexdespot_sink') 

    statistik_sink = Node(DataSink(base_directory=join(projectdir,'derivatives', 'statistik'),
                                    remove_dest_dir=True,
                                    parameterization=False),
                            name='statistik_sink') 

    # if data should be stored in subdirs with session_id make sure session_id is connected
    # input_names=["subject_id"] vs. input_names=["subject_id", "session_id"]
    out_dir = Node(Function(function=container_name,
                                       input_names=["subject_id", "session_id"],
                                       output_names=["dirname"]),
                            name='out_dir')

    rename_t1w = Node(Rename(format_string=("sub-%(subject_id)s_ses-%(session_id)s_space-pattmp_T1w"),
                                 keep_ext=True),
                          name='rename_t1w')

    rename_flair = Node(Rename(format_string=("sub-%(subject_id)s_ses-%(session_id)s_space-pattmp_FLAIR"),
                                 keep_ext=True),
                          name='rename_flair')

    rename_pd = Node(Rename(format_string=("sub-%(subject_id)s_ses-%(session_id)s_space-pattmp_PD"),
                                 keep_ext=True),
                          name='rename_pd') 
    rename_t1c = Node(Rename(format_string=("sub-%(subject_id)s_ses-%(session_id)s_space-pattmp_T1c"),
                                 keep_ext=True),
                          name='rename_t1c')                       
    rename_mwf = Node(Rename(format_string=("sub-%(subject_id)s_ses-%(session_id)s_space-pattmp_mwf"),
                                 keep_ext=True),
                          name='rename_mwf')

    rename_nawm = Node(Rename(format_string=("sub-%(subject_id)s_ses-%(session_id)s_space-pattmp_nawmmask"),
                                 keep_ext=True),
                          name='rename_nawm')
    rename_dawm = Node(Rename(format_string=("sub-%(subject_id)s_ses-%(session_id)s_space-pattmp_dawmmask"),
                                 keep_ext=True),
                          name='rename_dawm')
                          
    rename_meanst2l = Node(Rename(format_string=("sub-%(subject_id)s_ses-%(session_id)s_roimean-t2l_means"),
                                 keep_ext=True),
                          name='rename_meanst2l')
    rename_meansnawm = Node(Rename(format_string=("sub-%(subject_id)s_ses-%(session_id)s_roimean-nawm_means"),
                                 keep_ext=True),
                          name='rename_meansnawm')
    rename_meansdawm = Node(Rename(format_string=("sub-%(subject_id)s_ses-%(session_id)s_roimean-dawm_means"),
                                 keep_ext=True),
                          name='rename_meansdawm')

    wf.connect([(infosource, out_dir, [('subject_id','subject_id'),
                                       ('session_id','session_id')]),
                
                (applymask_t1w, rename_t1w, [('out_file', 'in_file')]),
                (infosource, rename_t1w, [('subject_id', 'subject_id'),
                                          ('session_id', 'session_id')]),
                (out_dir, brainex_sink, [('dirname', 'container')]),
                (rename_t1w, brainex_sink, [('out_file', 'anat.@t1')]),
                
                (applymask_flair, rename_flair, [('out_file', 'in_file')]),
                (infosource, rename_flair, [('subject_id', 'subject_id'),
                                          ('session_id', 'session_id')]),
                (rename_flair, brainex_sink, [('out_file', 'anat.@flair')]),
                
                (applymask_pd, rename_pd, [('out_file', 'in_file')]),
                (infosource, rename_pd, [('subject_id', 'subject_id'),
                                             ('session_id', 'session_id')]),
                (rename_pd, brainex_sink, [('out_file', 'anat.@pd')]),
                
                (applymask_t1c, rename_t1c, [('out_file', 'in_file')]),
                (infosource, rename_t1c, [('subject_id', 'subject_id'),
                                             ('session_id', 'session_id')]),
                (rename_t1c, brainex_sink, [('out_file', 'anat.@t1c')]),
                
                (applymask_mwf, rename_mwf, [('out_file', 'in_file')]),
                (infosource, rename_mwf, [('subject_id', 'subject_id'),
                                         ('session_id', 'session_id')]),
                (out_dir, brainexdespot_sink, [('dirname', 'container')]),
                (rename_mwf, brainexdespot_sink, [('out_file', '@mwf')]),
                
                (nawm_bin, rename_nawm, [('out_file', 'in_file')]),
                (infosource, rename_nawm, [('subject_id', 'subject_id'),
                                          ('session_id', 'session_id')]),
                (out_dir, segmentation_sink, [('dirname', 'container')]),
                (rename_nawm, segmentation_sink, [('out_file', 'anat.@nawm')]),
                
                (dawm, rename_dawm, [('rmsmdawm.bin_file', 'in_file')]),
                (infosource, rename_dawm, [('subject_id', 'subject_id'),
                                          ('session_id', 'session_id')]),
                (rename_dawm, segmentation_sink, [('out_file', 'anat.@dawm')]),
                
                (roimeans_t2l, rename_meanst2l, [('output', 'in_file')]),
                (infosource, rename_meanst2l, [('subject_id', 'subject_id'),
                                          ('session_id', 'session_id')]),
                (out_dir, statistik_sink, [('dirname', 'container')]),
                (rename_meanst2l, statistik_sink, [('out_file', '@t2l')]),
                
                (roimeans_nawm, rename_meansnawm, [('output', 'in_file')]),
                (infosource, rename_meansnawm, [('subject_id', 'subject_id'),
                                                ('session_id', 'session_id')]),
                (rename_meansnawm, statistik_sink, [('out_file', '@nawm')]),
                
                (roimeans_dawm, rename_meansdawm, [('output', 'in_file')]),
                (infosource, rename_meansdawm, [('subject_id', 'subject_id'),
                                             ('session_id', 'session_id')]),
                (rename_meansdawm, statistik_sink, [('out_file', '@dawm')]),
               ])        

      
    wf.run(plugin='MultiProc')


def main():
    import argparse
    import os
    from bids.layout import BIDSLayout
    from os.path import abspath, dirname, join
    parser = argparse.ArgumentParser(description="Register to MNI Space")
    parser.add_argument('-s' ,'--subject', type=str, nargs='*', default=None, help='subjects SPMS')
    parser.add_argument('-ss' ,'--session', type=str, nargs='*', default=None, help='0m, ... 87m')

    args = parser.parse_args()
    subjects = args.subject
    projectdir = dirname(dirname(abspath(__file__)))
    
    layout = BIDSLayout(projectdir, derivatives=True, ignore=['code', 'tmp'], validate=False)

    if subjects is None:
        subjects = layout.get_subjects()

    run_wf(layout, subjects)
    #run_wf(subjects)


if __name__ == '__main__':
    main()
