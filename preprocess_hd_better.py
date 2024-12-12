from random import shuffle
import pandas as pd
import os
from time import time
import numpy as np
import torch
import glob
import pydicom
import matplotlib.pyplot as plt
import json
from collections import Counter
from datetime import datetime
import heapq
from tqdm import tqdm

SAVE_DIR = "save_dir"

'''
Function that takes in a path to a dicom file and returns the acquisition time of it. 
'''
def get_acquisition_time(dicom_file):
    # Load the DICOM file
    dicom_data = pydicom.dcmread(dicom_file)
    
    # Get the acquisition time if it exists, defaulting to None if missing
    acquisition_time = getattr(dicom_data, 'AcquisitionTime', None)
    
    # Convert acquisition time to a proper datetime object for sorting, if it exists
    if acquisition_time:
        # DICOM AcquisitionTime is formatted as HHMMSS or HHMMSS.FFFFFF (with optional fractional seconds)
        try:
            acquisition_time_dt = datetime.strptime(acquisition_time.split('.')[0], '%H%M%S')
        except ValueError:
            acquisition_time_dt = None  # Handle any unexpected time format
    else:
        acquisition_time_dt = None
    
    return acquisition_time_dt

'''
Get the patient data of all level three patients, filtered on uni-lateral 
'''
def get_level_3_patients(clinical_data_dir : str):
    outcome_df = pd.read_excel(clinical_data_dir, sheet_name = "TCIA Outcomes Subset")
    clinical_df = pd.read_excel(clinical_data_dir, sheet_name = "TCIA Patient Clinical Subset")
    clinical_df = clinical_df[clinical_df["BilateralCa"] == 0]

    patients = set()

    min_age = clinical_df["age"].min()
    max_age = clinical_df["age"].max()

    patient_data = {}

    race_split = {
        "White": 0,
        "Black or African American" : 0,
        "Asian" : 0,
        "Native Hawaiian or Pacific Islander": 0, 
        "American Indian or Alaska Native": 0,
        "Others": 0, 
    }

    pcr_pos = 0


    for index, row in outcome_df.iterrows():
        patient_id = int(row['SUBJECTID'])
        # print("Patient ", patient_id)
        pcr_status = row['PCR']

        if pd.isna(pcr_status):
            print("happened for patient ", patient_id)
            continue

        pcr_status = int(pcr_status)

        clinical_features = clinical_df[clinical_df["SUBJECTID"] == patient_id]
        
        # skip since bilateral
        if len(clinical_features) < 1:
            continue

        raw_age : float = clinical_features["age"].iloc[0]
        age = round((raw_age - min_age) / (max_age - min_age), 2)

        race = int(clinical_features["race_id"].iloc[0])

        hr_her2_category = clinical_features["HR_HER2_CATEGORY"].iloc[0]

        if pd.isna(hr_her2_category):
            print("missing hr her 2 category", patient_id)
            continue

        hr_her2_category = int(hr_her2_category)
        hr, her2 = 0, 0
        if hr_her2_category == 1:
            hr = 1
        elif hr_her2_category == 2:
            her2 = 1
        else:
            assert hr_her2_category == 3

        if pd.isna(race):  # Use pd.isna to check for NaN
            race_encoded = [0, 0, 0, 0, 0, 1]
            race_split["Nan"] += 1
        elif race == 1:
            race_encoded = [1, 0, 0, 0, 0, 0]
            race_split["White"] += 1
        elif race == 3:
            race_encoded = [0, 1, 0, 0, 0, 0]
            race_split["Black or African American"] += 1
        elif race == 4:
            race_encoded = [0, 0, 1, 0, 0, 0]
            race_split["Asian"] += 1
        elif race == 5:
            race_encoded = [0, 0, 0, 1, 0, 0]
            race_split["Native Hawaiian or Pacific Islander"] += 1
        elif race == 6:
            race_encoded = [0, 0, 0, 0, 1, 0]
            race_split["American Indian or Alaska Native"] += 1
        elif race == 50 or race == 0:
            race_encoded = [0, 0, 0, 0, 0, 1]
            race_split["Others"] += 1
        else:
            raise ValueError

        # menopausal status data not available so n/a        
        menopausal_status = [0, 0, 1]

        '''
        Fetch info about unused but present features (ER, PR, Ki-67)
        '''
        er = int(clinical_features["ERpos"].iloc[0])
        pr = int(clinical_features["PgRpos"].iloc[0])
        
        
        # Combine the  pcr, HR, HER2, normalized age, race-encoded, menopausal_status, raw age, er, pr 
        patient_info = [pcr_status, hr, her2, age, race_encoded, menopausal_status, raw_age, er, pr]
        patient_data[patient_id] = patient_info
        patients.add(patient_id)

    print("this is patient data", len(patient_data))
    print("this is patient data", len(patients))
    return patient_data, patients

'''
Function that loads each image as dicom then immediately as numpy

'''
def save_npy(images, patient_number, timepoint, phase):
    assert phase == "pre" or phase == "first" or phase == "second"

    def load_dicom_as_array(image_path):
        dicom_data = pydicom.dcmread(image_path)
        return dicom_data.pixel_array
    try:
        volume = np.array([load_dicom_as_array(path) for path in images])
        save_path = os.path.join(SAVE_DIR,f"{patient_number}_t{timepoint}_{phase}.npy")
        np.save(save_path, volume)
    except IsADirectoryError:
        print("patient ", patient_number)
        print("timepoint ", timepoint)
        print("phase ", phase)
        print("files", images)
    
def save_as_dicom_and_npy(level_three_patient_set : set, root_dir: str, save_path : str):
    print("patient size", len(level_three_patient_set))
    '''
    A set of all patients that have a valid number of slices for all four phases. Here, valid means the number of slices exists in one of the above set of slices.
    '''
    patients_saved_with_all_phases = set()
    patients_with_t0_t1 = set()

    '''
    Go through all the patients and save the first post contrast uni-lateral cropped DCE MRI as 3d numpy files.
    '''
    for patient_folder in tqdm(sorted(glob.glob(root_dir + "/*"))): 
        patient_number = patient_folder.split("_")[-1]

        try:
            # Try converting patient_number to an integer
            patient_number = int(patient_number)
        except ValueError:
            # If conversion fails (not a valid integer), skip this patient
            print(f"Skipping {patient_folder}, invalid patient number.")
            continue
        
        if patient_number not in level_three_patient_set:
            continue

        timepoints = glob.glob(patient_folder + "/*")
        
        # print(timepoints)

        # Extract the date from each timepoint's folder name and sort based on the date
        def extract_date_from_timepoint(timepoint):
            # Assuming date is in MM-DD-YYYY format in the folder name
            last_part = timepoint.split('/')[-1]
    
            # Split this component by '-' and take the first three parts
            date_components = last_part.split('-')[:3]
            
            # Join these components to create the date string
            date_str = '-'.join(date_components)
            print(date_str)
            # date_str = "-".join(date_str)  # Join the parts to form the date string
            return datetime.strptime(date_str, "%m-%d-%Y")
        
        def count_subfolders(folder_path):
            # Count the number of subfolders inside the folder
            return len(glob.glob(folder_path + "/*"))
        
        # Sort timepoints based on the extracted date
        timepoints_sorted = sorted(timepoints, key=extract_date_from_timepoint)

        duplicates, times = set(), set()
        for timepoint in timepoints_sorted:
            time = extract_date_from_timepoint(timepoint)
            if time in times:
                duplicates.add(time)
            else:
                times.add(time)

        '''
        If there's more than one folder for any timepoint, drop the one that has fewer folders
        '''
        if len(duplicates) == 1:
            # Remove the folder that has the fewest subfolders
            min_subfolder = min([folder for folder in timepoints_sorted if extract_date_from_timepoint(folder) in duplicates], 
                                key=lambda x: count_subfolders(x))
            timepoints_sorted.remove(min_subfolder)
        elif len(duplicates) == 2:
            min_subfolders = heapq.nsmallest(2,[folder for folder in timepoints_sorted if extract_date_from_timepoint(folder) in duplicates], 
                                key=lambda x: count_subfolders(x))
            for folder in min_subfolders:
                timepoints_sorted.remove(folder)

        '''
        Skip if there's not exactly four folders for the four timepoints
        '''
        if len(timepoints_sorted) < 4:
            continue
        
        t_0_saved, t_1_saved, t_2_saved, t_3_saved = False, False, False, False

        for idx, timepoint in enumerate(timepoints_sorted[:4]):
            timepoint_folder = glob.glob(timepoint + "/*")
            ser_pe1_folders = [folder for folder in timepoint_folder if "SER" in folder or "PE1" in folder or "PE2" in folder]           
            
            cleared = False      
            for folder in timepoint_folder:
                if len(glob.glob(folder + "/*.dcm")) in [144, 150, 156, 168, 174, 180, 186, 192, 198, 204, 222, 234, 312]:
                    num_slices = int(len(glob.glob(folder + "/*.dcm")) / 3)
                    pre = sorted(glob.glob(folder + "/*.dcm"))[:num_slices]
                    first_post = sorted(glob.glob(folder + "/*.dcm"))[num_slices:num_slices*2]
                    second_post = sorted(glob.glob(folder + "/*.dcm"))[num_slices*2:]

                    save_npy(pre, patient_number=patient_number, timepoint=idx, phase="pre")
                    save_npy(first_post, patient_number=patient_number, timepoint=idx, phase="first")
                    save_npy(second_post, patient_number=patient_number, timepoint=idx, phase="second")

                    cleared = True
                elif len(glob.glob(folder + "/*.dcm")) in [220, 224, 232, 240, 248]:
                    num_slices = int(len(glob.glob(folder + "/*.dcm")) / 4)
                    pre = sorted(glob.glob(folder + "/*.dcm"))[:num_slices]
                    first_post = sorted(glob.glob(folder + "/*.dcm"))[num_slices:num_slices*2]
                    second_post = sorted(glob.glob(folder + "/*.dcm"))[num_slices*2:num_slices*3]
                    
                    save_npy(pre, patient_number=patient_number, timepoint=idx, phase="pre")
                    save_npy(first_post, patient_number=patient_number, timepoint=idx, phase="first")
                    save_npy(second_post, patient_number=patient_number, timepoint=idx, phase="second")

                    cleared = True
                elif len(glob.glob(folder + "/*")) in [300, 360, 390, 420, 450]:
                    num_slices = int(len(glob.glob(folder + "/*.dcm")) / 6)
                    pre, first_post, second_post = [], [],  []
                    for i, slice_file in enumerate(sorted(glob.glob(folder + "/*"))):
                        dynamic_index = i % 6
                        if dynamic_index == 1: 
                            pre.append(slice_file)
                        elif dynamic_index == 2:
                            first_post.append(slice_file)
                        elif dynamic_index == 3:
                            second_post.append(slice_file)
                    assert len(pre) == len(first_post) == len(second_post) == num_slices

                    save_npy(pre, patient_number=patient_number, timepoint=idx, phase="pre")
                    save_npy(first_post, patient_number=patient_number, timepoint=idx, phase="first")
                    save_npy(second_post, patient_number=patient_number, timepoint=idx, phase="second")

                    cleared = True

            if not cleared:
                # in this case, there's no folder with a recognized size, so get individual dynamics folders 
                if len(ser_pe1_folders) == 0:
                    # print("This patient doesnt have recognized DCE folder", patient_number, "at timepoint ", idx)
                    # Skip all patients that don't have SER or PE1/PE2 files.
                    pass
                else: 
                    num_slices = len(glob.glob(ser_pe1_folders[0] + "/*"))
                    assert num_slices == len(glob.glob(ser_pe1_folders[1] + "/*"))
                    assert len(ser_pe1_folders) == 2

                    non_ser_pe1_folder = [folder for folder in timepoint_folder if "SER" not in folder and "PE1" not in folder and "PE2" not in folder 
                                          and len(glob.glob(folder + "/*")) == num_slices]
                    
                    folders_with_acquisition_time = []

                    for folder in non_ser_pe1_folder:
                        # Get the first DICOM file in the folder
                        dicom_files = glob.glob(os.path.join(folder, "*.dcm"))
                        
                        if dicom_files:
                            first_slice = dicom_files[0]  # Access the first slice
                            acquisition_time = get_acquisition_time(first_slice)
                            
                            if dicom_files:
                                first_slice = dicom_files[0]  # Access the first slice
                                acquisition_time = get_acquisition_time(first_slice)
                                
                                if acquisition_time is not None:
                                    folders_with_acquisition_time.append((folder, acquisition_time))
                                else:
                                    print(f"No acquisition time found for the first slice in folder {folder}")
                            else:
                                print(f"No DICOM files found in folder {folder}")
                    
                    # Sort the list of folders based on acquisition time
                    # sorted_tuples is a tuple of (folder, acquisition time)
                    sorted_tuples = sorted(folders_with_acquisition_time, key=lambda x: x[1])
                    # Extract sorted folder paths
                    sorted_folders = [folder for folder, _ in sorted_tuples]
                    condition = True
                    if len(sorted_folders) < 3:
                        posts = [folder for folder in timepoint_folder if len(glob.glob(folder + "/*")) == num_slices * 2]
                        assert len(posts) == 1
                        posts = sorted(glob.glob(posts[0] + "/*.dcm"))

                        pre = sorted(glob.glob(sorted_folders[0] + "/*.dcm"))
                        first_post = posts[ : num_slices]
                        second_post = posts[num_slices : num_slices*2]
                    elif len(sorted_folders) == 3:
                        pre = sorted(glob.glob(sorted_folders[0] + "/*.dcm"))
                        first_post = sorted(glob.glob(sorted_folders[1] + "/*.dcm"))
                        second_post = sorted(glob.glob(sorted_folders[2] + "/*.dcm"))
                    elif len(sorted_folders) == 4:
                        if "SER" in ser_pe1_folders[0]:
                            name = ser_pe1_folders[0].split("SER")[0]
                            name = name.split("-")[-1].strip()
                        else:
                            name = ser_pe1_folders[1].split("SER")[0]
                            name = name.split("-")[-1].strip()

                        x =  [file for file in sorted_folders if name in file and "SUB" not in file]

                        pre = sorted(glob.glob(x[0] + "/*.dcm"))
                        first_post = sorted(glob.glob(x[1] + "/*.dcm"))
                        second_post = sorted(glob.glob(x[2] + "/*.dcm"))
                    else: 
                        if "SER" in ser_pe1_folders[0]:
                            name = ser_pe1_folders[0].split("SER")[0]
                            name = name.split("-")[-1].strip()
                        else:
                            name = ser_pe1_folders[1].split("SER")[0]
                            name = name.split("-")[-1].strip()
                            
                        x =  [file for file in sorted_folders if name in file and "SUB" not in file and "non fat" not in file
                              and "No Fat" not in file]
                        
                        pre = sorted(glob.glob(x[0] + "/*.dcm"))
                            
                        if "T1 Sagittal pre" in name:
                            assert len(x) == 1
                            others = [file for file in sorted_folders if "SUB" not in file and "non fat" not in file
                              and "No Fat" not in file and "pre" not in file]
                    
                            first_post = sorted(glob.glob(others[0] + "/*.dcm"))
                            second_post = sorted(glob.glob(others[1] + "/*.dcm"))
                        else:
                            if len(x) >= 3:
                                first_post = sorted(glob.glob(x[1] + "/*.dcm"))
                                second_post = sorted(glob.glob(x[2] + "/*.dcm"))
                            else:
                                condition = False
                    if condition:
                        assert len(pre) == len(first_post) == len(second_post)
                        save_npy(pre, patient_number=patient_number, timepoint=idx, phase="pre")
                        save_npy(first_post, patient_number=patient_number, timepoint=idx, phase="first")
                        save_npy(second_post, patient_number=patient_number, timepoint=idx, phase="second")
                        cleared = True
                    else:
                        cleared = False
                
                if cleared:
                    if idx == 0:
                        t_0_saved = True
                    elif idx == 1:
                        t_1_saved = True
                    elif idx == 2:
                        t_2_saved = True
                    else:
                        t_3_saved = True           
            else:
                if idx == 0:
                    t_0_saved = True
                elif idx == 1:
                    t_1_saved = True
                elif idx == 2:
                    t_2_saved = True
                else:
                    t_3_saved = True
                        
        if t_0_saved and t_1_saved and t_2_saved and t_3_saved:
            patients_saved_with_all_phases.add(patient_number)
        

    return patients_saved_with_all_phases
    
def create_json(patient_list, patient_data, output_path):
    patients = patient_list.copy()
    shuffle(patients)  

    data_dict = []
    for pid in patients:
        data = patient_data[pid]
        # [pcr_status, hr, her2, age, race_encoded, menopausal_status, raw age]
        
        #hr, her2, age, race_encoded, menopausal_status = data[1:6]
        non_mri_features = data[1:]

        data_dict.append({
        "patiend_id": pid, 
        "image_t0_pre": SAVE_DIR + "/" + str(pid) + "_t0_pre.npy",
        "image_t0_first": SAVE_DIR + "/" + str(pid) + "_t0_first.npy",
        "image_t0_second": SAVE_DIR + "/" + str(pid) + "_t0_second.npy",

        "image_t1_pre": SAVE_DIR + "/"  + str(pid) + "_t1_pre.npy",
        "image_t1_first": SAVE_DIR + "/" + str(pid) + "_t1_first.npy",
        "image_t1_second": SAVE_DIR + "/" + str(pid) + "_t1_second.npy",

        "image_t2_pre": SAVE_DIR + "/" + str(pid) + "_t2_pre.npy",
        "image_t2_first": SAVE_DIR + "/" + str(pid) + "_t2_first.npy",
        "image_t2_second": SAVE_DIR + "/" + str(pid) + "_t2_second.npy",

        "image_t3_pre": SAVE_DIR + "/" + str(pid) + "_t3_pre.npy",
        "image_t3_first": SAVE_DIR+ "/" + str(pid) + "_t3_first.npy",
        "image_t3_second": SAVE_DIR + "/" + str(pid) + "_t3_second.npy",
        "non_mri_features": non_mri_features,
        "label": int(data[0])})

    # Save the data dictionary as a JSON file
    with open(output_path, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)
    print("saved json file to ", output_path)


def get_patient_data(patient_data, saved_patients):
    pcr_positive, pcr_negative = 0, 0
    ages = []
    ages_pcr_pos = []
    ages_pcr_neg = []
    race_split = {
        "White": 0,
        "Black" : 0,
        "Asian" : 0,
        "Others": 0 
    }
    white_pos, black_pos, asian_pos, others_pos = 0, 0, 0, 0

    hr_pos, her2_pos = 0, 0
    hr_pos_pcr_pos = 0
    hr_neg_pcr_pos = 0

    her2_pos_pcr_pos = 0
    her2_neg_pcr_pos = 0

    # er_pos, er_neg, pr_pos, pr_neg = 0, 0, 0, 0
    print(len(saved_patients))
    for pid in saved_patients:
        features = patient_data[pid]
        pcr = features[0]
        
        hr = features[1]
        her2 = features[2]
        raw_age = features[6]
        race_encoded = features[4]
        menopausal_status = features[5]
        er = features[6]
        pr = features[7]

        if race_encoded == [1, 0, 0, 0, 0, 0]:
            race_split["White"] += 1
            if pcr:
                white_pos += 1
        elif race_encoded == [0, 1, 0, 0, 0, 0]:
            race_split["Black"] += 1
            if pcr:
                black_pos += 1
        elif race_encoded == [0, 0, 1, 0, 0, 0]:
            race_split["Asian"] += 1
            if pcr:
                asian_pos += 1
        else:
            race_split["Others"] += 1
            if pcr:
                others_pos += 1

        ages.append(raw_age)
        if hr:
            hr_pos += 1
        if her2:
            her2_pos += 1

        if pcr:
            pcr_positive += 1
            ages_pcr_pos.append(raw_age)
            if hr:
                hr_pos_pcr_pos += 1 
            else:
                hr_neg_pcr_pos += 1 
            if her2:
                her2_pos_pcr_pos += 1
            else: 
                her2_neg_pcr_pos += 1
        else:
            ages_pcr_neg.append(raw_age)
            pcr_negative += 1


    print("pCR positive count", pcr_positive)
    print("pCR negative count", pcr_negative)

    print("race split: ", race_split)

    print("white pcr pos", white_pos)
    print("black pcr pos", black_pos)
    print("asian pcr pos", asian_pos)
    print("others pcr pos", others_pos)

    ages = np.array(ages)
    mean_age = np.mean(ages)
    std_age = np.std(ages)

    ages_pcr_pos = np.array(ages_pcr_pos)
    ages_pcr_neg = np.array(ages_pcr_neg)

    print("age mean", mean_age)
    print("age std", std_age)

    print("PCR positive age mean", np.mean(ages_pcr_pos))
    print("PCR positive age std", np.std(ages_pcr_pos))
    print("PCR negative age mean", np.mean(ages_pcr_neg))
    print("PCR negative age std", np.std(ages_pcr_neg))


    print("Hr positive count", hr_pos)
    print("Hr negative count", len(saved_patients) - hr_pos)
    print("HR positive and pcr positive", hr_pos_pcr_pos)
    print("HR negative and pcr positive", hr_neg_pcr_pos)


    print("Her2 positive count", her2_pos)
    print("Her2 negative count", len(saved_patients) - her2_pos)
    print("her2 positive and pcr positive", her2_pos_pcr_pos)
    print("her2 negative and pcr positive", her2_neg_pcr_pos)


def main():
    '''
    READ IN THE IMAGES FROM ISPY1 AND SAVE THEM AS DICOM IMAGES
    '''
    ispy1_dir = "ISPY1"
    level_three_clinical_data = "clinical_data.xlsx"
    patient_data, patients = get_level_3_patients(level_three_clinical_data)
    
    print(len(patients))

    
    #all_four_timepoints = save_as_dicom_and_npy(level_three_patient_set=patients, root_dir=ispy1_dir, save_path=SAVE_DIR)
    # print("Cohort 1 number of patients saved with all four phases: ", len(all_four_timepoints), "out of ", len(patients), "patients")

    patient_set = []
  
    for image in sorted(glob.glob(os.path.join(SAVE_DIR, "*"))):
        if os.path.isfile(image):  # Ensure we're processing files, not directories
            # Extract patient ID by removing the 'save_dir' part of the file name
            pid = int(image.split("/")[-1].split("_")[0].replace("save_dir", ""))
            patient_set.append(pid)
    

    patient_count = Counter(patient_set)
    
    # Get a list of patient IDs that appear exactly four times
    saved_patients = [pid for pid, count in patient_count.items() if count == 4*3]
  
    get_patient_data(patient_data, saved_patients)
    # Get a list of patient IDs that appear exactly four times
    print("Number of patients saved with all four time points: ", len(saved_patients))
    
    '''
    Read in non-MRI data for all patients
    '''
    create_json(
        saved_patients,
        patient_data,
        output_path ="newjson.json"
    )

   
if __name__ == "__main__":
    main()