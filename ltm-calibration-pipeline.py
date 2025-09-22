from src.ltm_radcal import RadCal
from pathlib import Path
from natsort import natsorted

# UPDATE PATHS HERE:(otherwise, paths are assumed to be the following subfolders)
input_dir       = './ltm-raw-data/'
output_dir      = './ltm-calibrated-data/' 
quicklook_dir   = './ltm-quicklooks/'

# Initialize LTM calibration object:
radcal      = RadCal()
plt_details = False         # Flag to plot calibration products of calibration pipeline (for debugging)
overwrite   = False         # Flag to overwrite existing output files (if False, skips existing files)
iframe      = 0             # Frame index to plot if plt_details=True
mask_ind    = 350           # Mask index --> where masked pixels begin

# Create output directories if they don't exist:
output_dir    = Path(output_dir)
quicklook_dir = Path(quicklook_dir)
if output_dir is not None:
    output_dir.mkdir(parents=True, exist_ok=True)

if quicklook_dir is not None:
    quicklook_dir.mkdir(parents=True, exist_ok=True)

# List files in input directory: 
input_dir = Path(input_dir)
file_list = natsorted(list(input_dir.glob('*.h5')) + list(input_dir.glob('*.hdf5')))
print(f'Found {len(file_list)} files in {input_dir}\n')

# Loop through files and run calibration pipeline:
for file_path in file_list:
    print(f'Processing file: {file_path.name}')
    
    if 'ltm-cal-ext' in file_path.name:
        calibration_flag = True
    else:
        calibration_flag = False

    # Load data from HDF5 file:
    radcal.load_data(file_path, calibration_flag=calibration_flag)

    # Apply row correction:
    radcal.apply_row_correction(mask_ind=mask_ind)

    # Calculated calibrated radiances:
    radcal.calc_calibrated_radiance(calibration_flag=calibration_flag, plot_gain_offset=plt_details)

    if plt_details:
        print("Plotting Internal Calibration Looks...")
        radcal.plot_corrected_data('Int', iframe)  

        print("Plotting Cold Space Calibration Looks...")
        radcal.plot_corrected_data('Cold', iframe)  

        print("Plotting Scene Look...")
        radcal.plot_corrected_data('Hot', iframe)  

        print("Plotting Calibrated Radiances...")
        radcal.plot_calibrated_radiance(iframe, plot_log=True)

    # Save calibrated data to output directory
    if output_dir is not None:
        output_file = output_dir / f'calibrated-{file_path.name}'
        radcal.save_calibrated_data(output_file, overwrite=overwrite)

    # Save quicklook plots to quicklook directory
    if quicklook_dir is not None:
        quicklook_file = quicklook_dir / f'{file_path.stem}-radiance.pdf'
        radcal.plot_quicklook(quicklook_file)
        print(f'Saved quicklook plot to: {quicklook_file}')

    print("\n"+"-"*50+"\n")