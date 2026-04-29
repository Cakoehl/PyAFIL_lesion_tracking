# PyAFIL lesion tracking

This code facilitates the analysis of lesions that intersect spatially within a time series. A prerequisite is that the MRIs of the time-series are aligned for example in a patient template space. 

The workflow of PyAFIL is illustrated in the following steps. 

<img width="450" alt="PyAFIL_workflow" src="https://github.com/user-attachments/assets/85688710-4bb6-4171-a76b-eb102a5d9c6e" />

### 0) **Segmented binary lesion masks**
Pre-segmented lesion masks were used as input files for the AFIL algorithm.

These had to be stored in a patient-specific template space.  
This meant that the data had to be prepared so that the image series and lesion masks were identically aligned for longitudinal analysis.

### 1) **Lesion density mask (LDM)**
A) Binary lesion masks from a patient’s time series are summed
All lesion masks from the time series were summed and stored in a **Lesion Density Mask (LDM)**.

- This reference mask contained all lesions from the entire time series.  
- It shows how frequently a voxel was segmented as a lesion (lesion density).  
- The LDM corresponds to the maximum spatial extent of the lesion.

B) Creation of Global Lesion Labels
The LDM was binarized 

C) binarized LDM get labeled
- Contiguous voxels in the binarized LDM received a number (global reference lesion label, **glbl**)  
- These labels were stored in the global reference mask (**gRM**)  
- The number of glbls corresponds to the number of individual lesions per patient  
- Lesions that merged over time received the same glbl (conglomerates share a label)

### 2) **Interim Lesion Labels**
Lesion masks at each time point were automatically labeled and assigned an **interim label (ilbl)**.

- Lesions may change or appear over time  
- Therefore, lesion labels are not consistent across time points  
- Automatic labeling assigns new identifiers at each time point  

### 3) **Creation of global lesion masks**
To ensure consistent lesion tracking over time:

- Spatial overlap between global labels (LDM) and interim lesion masks was computed  
- Lesions were relabeled using their corresponding global reference labels  
- This ensures identical lesion IDs across all time points  

### **Conglomerate lesions**
A dedicated tracking approach was used for merged lesions.

- Tracks predecessors and successors of each lesion  
- Identifies when lesions merge (“confluent event time”)  
- Allows individual tracking until lesion fusion occurs  
- Based on spatial overlap across consecutive time points  

### **Reading the MWF values**
MWF values were extracted by overlaying imaging data:

- The MWF map was overlaid with the global lesion masks  
- Values were extracted per lesion per time point  

For each lesion, the following parameters were computed:
- Volume  
- MWF (mean, median, minimum, maximum)  

These outputs were exported to Excel for longitudinal statistical analysis under /derivatives/lesion_stats.

## USAGE
First check the config file and the bids route. Also bids dataset_description must be available.  
-c ... config_file 

-s ... bids subject_id 

-f ... force re-run AFIL  

1) open the Terminal
2) You should run the code from the code directory. cd code 

3) python3 run_afil.py -c /code/afil/AFIL_config_bids.json -s DEV027 -f --> re-run a specific subject

   or

   python3 run_afil.py -c /code/afil/AFIL_config_bids.json --> will run all subjects found by bids

## DEMO

Click the button below to be redirected to the executable Jupyter Notebook file PyAFIL_demo.ipynb on mybinder.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Cakoehl/PyAFIL_lesion_tracking/HEAD?urlpath=%2Fdoc%2Ftree%2FPyAFIL_demo.ipynb)
