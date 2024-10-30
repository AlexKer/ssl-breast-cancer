"""
The preprocessed data follow structure:
ISPY1_sub
----/ISPY1_1001
--------/T1
------------dcm files
--------/T2
--------/T3
--------/T4
----/ISPY1_1002
....

My code is stupid, it only contains subjects that have 4 time stage and >=180 dcm files in each time stage, 
which means every subject which contains <60 dcm in their PE/SER folders will be skipped.
This preprocessing is just for the first stage of this project, and I will improve it in the future.

The preprocessed data just have 81 subjects
"""


import os
import shutil
import numpy as np
from tqdm import tqdm
from datetime import datetime

original_dir = "ISPY1"
output_dir = "ISPY1_sub"

os.makedirs(output_dir, exist_ok=True)
errors = 0
skip_subjects = []
error_subjects = []


for subject_folder in tqdm(sorted(os.listdir(original_dir)), desc="Subjects"):
    subject_path = os.path.join(original_dir, subject_folder)
    
    if os.path.isdir(subject_path):
        date_folders = sorted(
            [f for f in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, f))],
            key=lambda date: datetime.strptime(date[:10], "%m-%d-%Y")
        )

        if len(date_folders) < 4:
            skip_subjects.append(subject_folder)
            continue
        
        if (len(date_folders) > 4):
            for i in range(len(date_folders)):
                if date_folders[i][:10] == date_folders[i+1][:10]:
                    destination_dir = os.path.join(subject_path, date_folders[i])
                    source_dir = os.path.join(subject_path, date_folders[i+1])
                    for f in os.listdir(source_dir):
                        shutil.move(os.path.join(source_dir, f), os.path.join(destination_dir, f))
                    shutil.rmtree(source_dir)
                    date_folders.pop(i+1)
                    break
        
        output_subject_path = os.path.join(output_dir, subject_folder)
        os.makedirs(output_subject_path, exist_ok=True)

        for i, date_folder in enumerate(date_folders):
            date_folder_path = os.path.join(subject_path, date_folder)
            sub_subfolders = [f for f in os.listdir(date_folder_path) if os.path.isdir(os.path.join(date_folder_path, f))]

            find = False
            for f in sub_subfolders:
                images = os.listdir(os.path.join(date_folder_path, f))
                if len(images) >= 180:
                    target_subfolder = os.path.join(date_folder_path, f)
                    find = True
                    break
            if not find:
                #print(f"Skipping {subject_folder}: {date_folder_path}, as it does not contain 180 images.")
                error_subjects.append(subject_folder)
                continue

            output_t_folder = os.path.join(output_subject_path, f"T{i+1}")
            os.makedirs(output_t_folder, exist_ok=True)

            for img_index in range(60, 120):
                src_image_path = os.path.join(target_subfolder, images[img_index])
                dst_image_path = os.path.join(output_t_folder, images[img_index])
                shutil.copy2(src_image_path, dst_image_path)


print(f"Total {len(np.unique(error_subjects))} subjects has been skipped due to missing images. They are:\n {np.unique(error_subjects)}")
print(f"There are {len(skip_subjects)} subjects has been skipped, they are:\n {skip_subjects}")

original_dir = "ISPY1_sub"
for subject_folder in tqdm(sorted(os.listdir(original_dir)), desc="Subjects"):
    subject_path = os.path.join(original_dir, subject_folder)

    if len(os.listdir(subject_path)) < 4:
        shutil.rmtree(subject_path)

print(len(os.listdir(original_dir)))

