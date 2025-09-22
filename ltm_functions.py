import h5py    
import numpy as np    
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle

def print_dict_structure(data, indent=0):
    # Check if the data is a dictionary
    if isinstance(data, dict):
        for key, value in data.items():
            # Print the current key with indentation
            print(' ' * indent + f"Group/Subgroup: {key}")
            
            # If the value is another dictionary, recursively print its structure
            print_dict_structure(value, indent + 2)  # Increase indent for nested groups
    else:
        # If it's not a dictionary, just print the data type (or shape for arrays)
        print(' ' * indent + f"Data type: {type(data)}")

    return 


def decimate_tdi_frame(channel_map_all, data):
    from matplotlib import colormaps
    
    # Compute the gradient along the columns to detect transitions between channels and take abs to get clear edges
    gradient = np.gradient(channel_map_all, axis=1)  
    gradient_magnitude = np.abs(gradient)
    
    plt.figure(figsize=(6, 4), dpi=100)
    im1 = plt.imshow(gradient_magnitude, aspect='auto', cmap='plasma', interpolation=None)
    plt.title("Gradient Magnitude: \nLooking for channel boundaries using channel map") 
    plt.colorbar(im1)
    plt.show()
    
    # Use a threshold to detect where significant transitions happen (90%) & create a mask
    threshold = np.percentile(gradient_magnitude, 90)  
    transition_mask = gradient_magnitude > threshold
    transition_mask_1D = np.mean(transition_mask, axis=0)
    x = np.where(transition_mask_1D==0)[0]
    
    plt.figure(figsize=(6, 4), dpi=100)
    im2 = plt.imshow(transition_mask, aspect='auto', cmap='plasma', interpolation=None)
    plt.title("Mask of Channel Boundaries") 
    plt.colorbar(im2)
    plt.show()
    
    # Find channel widths: (i.e., where transitions are detected)
    channel_widths = []
    start = x[0]
    
    # If the sequence breaks add the end index (check the length is greater than 2)
    for i in range(1, len(x)):
        if x[i] != x[i - 1] + 1:
                channel_widths.append((start, x[i - 1]))  
                start = x[i]
            
    # Add the last run
    channel_widths.append((start, x[-1]))
    
    # Discard channels where the difference between start and end is 1
    channel_widths = [run for run in channel_widths if run[1] - run[0] > 1]
    
    plt.figure(figsize=(6, 4), dpi=100)
    plt.title("1D Mask and channel widths")
    colors  = colormaps.get_cmap("tab20")
    
    plt.plot(transition_mask_1D)
    for idx, (start, end) in enumerate(channel_widths):
        plt.plot([start, end], [transition_mask_1D[start], transition_mask_1D[end]], marker='o', color=colors(idx), markersize=5, label=f"Run {idx+1}: {start}-{end}")
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    
    # Create TDI frames of data by suming within the channels:
    summed_data = np.zeros((data.shape[0], len(channel_widths)))
    
    for i, (start_col, end_col) in enumerate(channel_widths):
        summed_data[:, i] = np.sum(data[:, start_col:end_col], axis=1)

    # To match LTM GDS pipeline outputs transpose axis and add 1 at the end: (ASK HENRY WHY)???
    summed_data = np.vstack([summed_data, np.ones((1, summed_data.shape[1]))])
    summed_data = summed_data.T
    
    plt.figure(figsize=(6, 4), dpi=100)
    im3 = plt.imshow(summed_data, aspect='auto', cmap='plasma', interpolation=None)
    plt.title("TDI Adjusted Summed Data") 
    plt.colorbar(im3)
    plt.show()

    return summed_data
    

def import_caltest_datatables():
    # Parse Calibration data from RadCal_Index (Only really need to do this once!)
    mat_data = sio.loadmat('./calibration-tables/RadCalIndex.mat')
    
    temperature_gain = mat_data['RadCal_Index']['c2'][0,0]
    
    channel_map = mat_data['RadCal_Index']['ChannelMap'][0,0]
    channel_map_mean = mat_data['RadCal_Index']['ChannelMapMean'][0,0]
    channel_map_all = mat_data['RadCal_Index']['ChannelMapAll'][0,0]
    
    stray_light = mat_data['RadCal_Index']['StrayLight'][0,0]
    stray_light_mean = mat_data['RadCal_Index']['StrayLightMean'][0,0]
    stray_light_mask = mat_data['RadCal_Index']['StrayLightFactorMask'][0,0]
    
    bad_pixel_table = mat_data['RadCal_Index']['badPixelTable'][0,0]
    
    # Change temperature gain in channel x row format: 
    temperature_gain_cal = temperature_gain * channel_map_all
    temperature_gain_cal_tdi = decimate_tdi_frame(channel_map_all, temperature_gain_cal)
    
    stray_light_correction = stray_light_mean * channel_map_all
    stray_light_correction_tdi = decimate_tdi_frame(channel_map_all, stray_light_correction)
    
    # Save pickle files: 
    files_and_data = [('./calibration-tables/TemperatureGain_TDI.pkl', temperature_gain_cal_tdi), 
                      ('./calibration-tables/TemperatureGain.pkl', temperature_gain_cal),
                      ('./calibration-tables/StrayLightCorr_TDI.pkl', stray_light_correction_tdi),
                      ('./calibration-tables/StrayLightCorr.pkl', stray_light_correction)
                     ]
    
    # Loop through the list of tuples with enumerate
    for pickle_file_path, pickle_data in files_and_data:
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(pickle_data, pickle_file)
    
    # For visualization:
    fig, axes = plt.subplots(1, 6, figsize=(20, 4), dpi=200)  
    im0 = axes[0].imshow(channel_map_all, aspect='auto', cmap='plasma')
    axes[0].set_title("Channel Map (Averaged)")
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(temperature_gain, aspect='auto', cmap='plasma')
    axes[1].set_title("Temperature Gain")
    fig.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(channel_map_all * temperature_gain, aspect='auto', cmap='plasma')
    axes[2].set_title("Temperature Gain * Channel Map")
    fig.colorbar(im2, ax=axes[2])
    
    im3 = axes[3].imshow(stray_light_mean, aspect='auto', cmap='plasma')
    axes[3].set_title("Stray Light Mean")
    fig.colorbar(im3, ax=axes[3])
    
    im4 = axes[4].imshow(stray_light_mask, aspect='auto', cmap='plasma')
    axes[4].set_title("Stray Light Mask")
    fig.colorbar(im4, ax=axes[4])
    
    im5 = axes[5].imshow(stray_light_mask * stray_light_mean, aspect='auto', cmap='plasma')
    axes[5].set_title("Stray Light Mask * Mean")
    fig.colorbar(im5, ax=axes[5])
    
    plt.tight_layout()
    plt.show()
    
    return
