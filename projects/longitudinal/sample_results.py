import os
import shutil

# Define the methods and Rs
methods = ("Baseline_Souza_IKIK", "Baseline_Ours_E2EVarNet", "Enhanced_Souza_UNet", "Enhanced_Ours_Transformers")#, "Enhanced_Ours_UNet", 'Enhanced_Trained_and_tested_on_the_atlas','Enhanced_Trained_on_previous_scan_and_tested_on_the_atlas')
# methods = ("Baseline_Ours_E2EVarNet", "Enhanced_Ours_Transformers")
Rs = ("R5", "R10", "R15", "R20")
files = ('preds',)
subjs = ('e13991s3_P01536' ,'e13993s4_P16896','e14078s3_P02048', 'e14079s3_P09216', 'e14080s3_P18944', 'e14081s3_P25600')
dist = 'sample_results_v_6_sep'
# Loop through each method and R
for subj in subjs:
    for method in methods:
        for R in Rs:
            for file in files:
            # Construct the path to the CSV file
                src_path = f"/home/amirmohammad.shamaei/imri_result/{R}/{method}/segment/{subj}.7_{file}_synthseg.nii"
                dst_path = f"/home/amirmohammad.shamaei/imri_result/{dist}/"
                # Check if the destination directory exists
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)

                # Copy the file
                shutil.copy(src_path, dst_path+f'{R}_{method}_{subj}_{file}_seg.nii')

for subj in subjs:
    for method in methods:
        for R in Rs:
            for file in files:
            # Construct the path to the CSV file
                src_path = f"/home/amirmohammad.shamaei/imri_result/{R}/{method}/{subj}.7_{file}.nii"
                dst_path = f"/home/amirmohammad.shamaei/imri_result/{dist}/"
                # Check if the destination directory exists
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)

                # Copy the file
                shutil.copy(src_path, dst_path+f'{R}_{method}_{subj}_{file}.nii')

Rs = ("R5",)
files = ('targets',)
methods = ("Baseline_Souza_IKIK",)
subjs = ('e13991s3_P01536','e13993s4_P16896','e14078s3_P02048', 'e14079s3_P09216', 'e14080s3_P18944', 'e14081s3_P25600')
# Loop through each method and R
for subj in subjs:
    for method in methods:
        for R in Rs:
            for file in files:
            # Construct the path to the CSV file
                src_path = f"/home/amirmohammad.shamaei/imri_result/{R}/{method}/segment/{subj}.7_{file}_synthseg.nii"
                dst_path = f"/home/amirmohammad.shamaei/imri_result/{dist}/"
                # Check if the destination directory exists
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)

                # Copy the file
                shutil.copy(src_path, dst_path+f'{R}_{method}_{subj}_{file}_seg.nii')

for subj in subjs:
    for method in methods:
        for R in Rs:
            for file in files:
            # Construct the path to the CSV file
                src_path = f"/home/amirmohammad.shamaei/imri_result/{R}/{method}/{subj}.7_{file}.nii"
                dst_path = f"/home/amirmohammad.shamaei/imri_result/{dist}/"
                # Check if the destination directory exists
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)

                # Copy the file
                shutil.copy(src_path, dst_path+f'{R}_{method}_{subj}_{file}.nii')