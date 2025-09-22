import h5py    
import numpy as np    
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle
import warnings

from dataclasses import dataclass, field
from ltm_model import LTM_Model

@dataclass
class RadCal:
    """Class to handle loading, storing & processing LTM calibration and observation data."""

    # Initialize the fields to store calibration data (raw data for each group and look)
    Int: dict = field(default_factory=dict)
    Cold: dict = field(default_factory=dict)
    Hot: dict = field(default_factory=dict)
    
    # Store the corrected data
    row_corrected_data: dict = field(default_factory=dict)

    # Metadata storage
    times: dict = field(default_factory=dict)
    tdi_flg: dict = field(default_factory=dict)
    
    # Temperature storage
    temps: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize nested dictionaries for metadata and corrected data."""
        self.row_corrected_data = {k: {} for k in ['Int', 'Cold', 'Hot']}
        self.times = {k: {} for k in ['Int', 'Cold', 'Hot']}
        self.tdi_flg = {k: {} for k in ['Int', 'Cold', 'Hot']}
        self.temps = {k: {} for k in ['Int', 'Cold', 'Hot']} 

    def assert_array_shape(self, array, calibration_file=False):
        """Asserts that the given array has a shape of (frames, channels/rows, columns) where:
            - Frames can be any size (first dimension is flexible)
            - Second dimension must be either 15 (channels) or 288 (rows)
            - Third dimension must be either 384 or 385 (columns)

        Parameters:
            array (np.ndarray): Input array to validate.

        Raises:
            ValueError: If the array does not match the expected shape constraints.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a NumPy array.")

        if array.ndim != 3 and not calibration_file:
            raise ValueError(f"Expected a 3D array, but got {array.ndim}D.")

        if calibration_file:
            if array.ndim != 2:
                raise ValueError(f"Expected a 2D array for calibration file, but got {array.ndim}D.")
            
            second_dim, third_dim = array.shape
        else: 
            frames, second_dim, third_dim = array.shape

        if second_dim not in (15, 288):
            raise ValueError(f"Second dimension must be 15 (channels) or 288 (rows), but got {second_dim}.")

        if third_dim not in (384, 385):
            raise ValueError(f"Third dimension must be 384 or 385 (columns), but got {third_dim}.")

        return 

    def load_data(self, file_path: str, calibration_flag: bool = False):
        """Load the HDF5 file and extract data into RadCal object."""
        
        with h5py.File(file_path, 'r') as f:
            if calibration_flag:
                # File with name ltm-cal-ext-*.h5 from LTM FM Calibration Campaign: 
                print(f"Loading calibration data from {file_path}")

                # Define dataset mappings
                mapping = {
                    "Int": {"bb_look11": "frames"},
                    "Cold": {"cs_look11": "frames"},
                    "Hot": {"scene": "frames"}
                }

                # Loop through `Int`, `Cold`, `Hot` categories --> Format (frames, rows, columns)
                for category, datasets in mapping.items():
                    if category in f:
                        for key, dataset_name in datasets.items():
                            if dataset_name in f[category]:
                                dataset =  f[category][dataset_name][:].transpose(0, 1, 2)
                                self.assert_array_shape(dataset)
                                self.__dict__[category][key] = dataset
                                
                            else:
                                print(f"Warning: {dataset_name} missing in {category}!")
                        
                        print(f"Original {category} -> {key}: shape {f[category][dataset_name][:].shape}")
                        print(f"Loaded {category} -> {key}: shape {self.__dict__[category][key].shape}")

                # Extract times & temperatures:
                for category, subgroups in mapping.items():
                    if category in f:
                        for subgroup_name, dataset_name in subgroups.items():
                            # Extract times, temperatures, and tdi flags for each subgroup
                            self.times[category][subgroup_name] = f[category]['startTime'][()] if 'startTime' in f[category] else None
                            self.temps[category][subgroup_name] = f[category]['SceneTemp'][()] if 'SceneTemp' in f[category] else None
                            self.tdi_flg[category][subgroup_name] = f[category]['tdi'][()] if 'tdi' in f[category] else False
                    else:
                        print(f"Warning: {category} missing in file!")

            else: 
                print(f"Loading observation data from {file_path}")

                # Extract and structure the calibration data
                calibration_group = f['calibration']
                self.Int['bb_look11'] = calibration_group['bb_look11'][:]
                self.Int['bb_look21'] = calibration_group['bb_look21'][:]
                self.Cold['cs_look11'] = calibration_group['cs_look11'][:]
                self.Cold['cs_look21'] = calibration_group['cs_look21'][:]

                # Assert files are expected size: 
                self.assert_array_shape(self.Int['bb_look11'])
                self.assert_array_shape(self.Int['bb_look21'])
                self.assert_array_shape(self.Cold['cs_look11'])
                self.assert_array_shape(self.Cold['cs_look21'])

                # Access the observation group:
                observation_group = f['observation']
                for subgroup in observation_group:
                    self.Hot[subgroup] = observation_group[subgroup][:]
                    self.assert_array_shape(self.Hot[subgroup])
                    print(f"Loaded subgroup: {subgroup}")

                # Loop through the subgroups within each group and save attributes:
                for key in ['calibration', 'observation']:
                    for subgroup in f[key]:
                        # print(f"{key}/{subgroup} attributes: {list(f[key][subgroup].attrs.keys())}")
                        # print(f"{key}/{subgroup} attributes: {(f[key][subgroup].attrs['time'])}")
                        if key == 'calibration' and 'bb' in subgroup:
                            self.times['Int'][subgroup] = f[key][subgroup].attrs['time']
                            self.tdi_flg['Int'][subgroup] = f[key][subgroup].attrs['tdi']
                            self.temps['Int'][subgroup] = f[key][subgroup].attrs.get('SceneTemp', 290)

                        elif key == 'calibration' and 'cs' in subgroup:
                            self.times['Cold'][subgroup] = f[key][subgroup].attrs['time']
                            self.tdi_flg['Cold'][subgroup] = f[key][subgroup].attrs['tdi']
                            self.temps['Cold'][subgroup] = f[key][subgroup].attrs.get('SceneTemp', 5.0)

                        else: 
                            self.times['Hot'][subgroup] = f[key][subgroup].attrs['time']
                            self.tdi_flg['Hot'][subgroup] = f[key][subgroup].attrs['tdi']
                            self.temps['Hot'][subgroup] = f[key][subgroup].attrs.get('SceneTemp', None)

            # Verification:
            # print(f"Time times for Int: {self.times['Int']}")
            # print(f"Time times for Cold: {self.times['Cold']}")
            # print(f"Time times for Hot: {self.times['Hot']}")
            print(f"Data successfully loaded from {file_path}")
            
            return 
    
    def load_temperature_gain_table(self, tdi_flag: bool = True):
        # Temperature dependent fit of detector:                
        if tdi_flag:
            with open('./calibration-tables/TemperatureGain_tdi.pkl', 'rb') as f:
                b2 = pickle.load(f)
    
        else:
            with open('./calibration-tables/TemperatureGain.pkl', 'rb') as f:
                b2 = pickle.load(f).T
                
        self.assert_array_shape(b2, calibration_file=True)      
        print(f'Shape of temperature dependent gain table: {b2.shape}')

        return b2

    def load_channel_map(self):
        mat_data = sio.loadmat('./calibration-tables/RadCalIndex.mat')
        channel_map = mat_data['RadCal_Index']['ChannelMap'][0,0].T
        self.assert_array_shape(channel_map, calibration_file=True)

        return channel_map

    def load_bad_pixel_table(self):
        print("Still needs to be added")
        return

    def apply_row_correction(self, mask_ind: int = 365):
        """Apply row correction to the 'Int', 'Cold', and 'Hot' data."""

        for key in ['Int', 'Cold', 'Hot']:
            for subgroup in getattr(self, key):
                
                # Temperature dependent fit of detector:                
                # b2 = self.load_temperature_gain_table(self, tdi_flag=self.tdi_flg[key][subgroup]) --> why doesn't work? 
                if self.tdi_flg[key][subgroup]:
                    with open('./calibration-tables/TemperatureGain_tdi.pkl', 'rb') as f:
                        b2 = pickle.load(f)
            
                else:
                    with open('./calibration-tables/TemperatureGain.pkl', 'rb') as f:
                        b2 = pickle.load(f).T
                        
                self.assert_array_shape(b2, calibration_file=True)      
                print(f'Shape of temperature dependent gain table: {b2.shape}')
            
                # Get the (frames, channels, rows) array for each look:
                data = getattr(self, key)[subgroup]
                num_frames = data.shape[0]
                
                # Initialize the corrected data structure for the subgroup if it does not exist
                if subgroup not in self.row_corrected_data[key]:
                    self.row_corrected_data[key][subgroup] = {}

                offset_avg_list = []
                cor_frames_list = []
                stable_cor_frames_list = []
                stray_light_cor_frames_list = []

                # Get the avg value over all masked pixels during internal calibration at the start of the observation (averaging all frames): 
                avg_int_offset = np.mean(self.Int['bb_look11'][:, :, mask_ind:-1])

                for iframe in range(num_frames):
                    frame_data = data[iframe, :]
                    
                    # Apply the row & offset correction calculation:
                    offset_avg = self.row_cor_offset(frame_data, mask_ind)
                    cor_frames = self.row_cor(frame_data, mask_ind, avg_int_offset)
                    stable_cor_frames =  self.row_cor(frame_data, mask_ind, avg_int_offset, gain_ratio=b2)
                    
                    # Stray light correction:
                    stray_light_cor_frames = self.stray_light_correction(stable_cor_frames, key, subgroup)
                    
                    offset_avg_list.append(offset_avg)
                    cor_frames_list.append(cor_frames)
                    stable_cor_frames_list.append(stable_cor_frames)
                    stray_light_cor_frames_list.append(stray_light_cor_frames)

                # Store the results in the row_corrected_data dictionary
                self.row_corrected_data[key][subgroup]['rawdata'] = np.array(data)
                self.row_corrected_data[key][subgroup]['offsetAverage'] = np.array(offset_avg_list)
                self.row_corrected_data[key][subgroup]['corFrames'] = np.array(cor_frames_list)
                self.row_corrected_data[key][subgroup]['stableCorFrames'] = np.array(stable_cor_frames_list)
                self.row_corrected_data[key][subgroup]['stableRowSLCorFrames'] = np.array(stray_light_cor_frames_list)

    def row_cor(self, M: np.ndarray, mask_ind: int, avg_int_offset: float, gain_ratio: np.ndarray = None) -> np.ndarray:
        """Subtracting the mean of rows mask_ind:-1 from the rest of the frame."""
       
        # If gain_ratio is not provided, create a default one (array of ones with the same shape as M)
        if gain_ratio is None:
            gain_ratio = np.ones_like(M)

        # Ensure gain_ratio has the same shape as M
        if gain_ratio.shape[1] != M.shape[1]:
            raise ValueError("gain_ratio must have the same shape as M")
        
        return M - gain_ratio * (np.mean(M[:, mask_ind:-1], axis=1) - avg_int_offset)[:, np.newaxis]

    def row_cor_offset(self, M: np.ndarray, mask_ind: int) -> np.ndarray:
        """Calculates the average value of masked pixels in given row."""
        
        return np.mean(M[:, mask_ind:-1], axis=1)

    def stray_light_correction(self, M: np.ndarray, key, subgroup):
        # Get stray light measurement: 
        if self.tdi_flg[key][subgroup]:
            with open('./calibration-tables/StrayLightCorr_tdi.pkl', 'rb') as f:
                measured_straylight = pickle.load(f)
        else:
            with open('./calibration-tables/StrayLightCorr.pkl', 'rb') as f:
                measured_straylight = pickle.load(f).T
                
        self.assert_array_shape(measured_straylight, calibration_file=True)   

        return M - measured_straylight

    def calc_calibrated_radiance(self, model: LTM_Model, calibration_flag=False):
        """Calibrate the radiance using the provided LTM model and temperatures."""
        # Get the times from the RadCal object for the calibration views
        t1 = self.times['Int']['bb_look11']
        t2 = self.times['Cold']['cs_look11']
        t4 = self.times['Int'].get('bb_look21', t1)   # Use t1 if t4 is missing (but check not using this)
        t5 = self.times['Cold'].get('cs_look21', t2)  # Use t2 if t5 is missing

        # Define the temperatures (in Kelvin)
        T1 = self.temps['Int']['bb_look11']
        T2 = self.temps['Cold']['cs_look11']
        T4 = self.temps['Int'].get('bb_look21', T1) 
        T5 = self.temps['Cold'].get('cs_look21', T2)

        # Calculate the radiance values for each temperature (T1, T2, T4, T5) using the LTM model:
        # opted to not add stray light here because I removed it earlier than in the pipeline figure 
        # (right after row correction) so theoretically if everything is liner order doens't matter 
        # and the frames I comparing to have already removed stray light contribution. --> Confirm with Rory! 
        B1 = model.calcRadiance(model.channelFilterMap, [T1]) 
        B2 = model.calcRadiance(model.channelFilterMap, [T2]) 
        B4 = model.calcRadiance(model.channelFilterMap, [T4]) 
        B5 = model.calcRadiance(model.channelFilterMap, [T5])

        # Check if you have calibration veiws before & after data (but code should have failed by now if you didn't..)
        if not calibration_flag:
            if 'cs_look11' not in self.temps['Int']:
                raise TypeError("Missing calibration view: cs_look11.")

            if 'cs_look21' not in self.temps['Int']:
                raise TypeError("Missing calibration view: cs_look21.")

            if 'bb_look11' not in self.temps['Int']:
                raise TypeError("Missing calibration view: bb_look11.")
                
            if 'bb_look21' not in self.temps['Int']:
                raise TypeError("Missing calibration view: bb_look21.")
            
            # Get the row corrected frames for each view (V1, V2, V4, V5)
            V1 = self.row_corrected_data['Int']['bb_look11']['stableCorFrames']
            V2 = self.row_corrected_data['Cold']['cs_look11']['stableCorFrames']
            V4 = self.row_corrected_data['Int']['bb_look21']['stableCorFrames']
            V5 = self.row_corrected_data['Cold']['cs_look21']['stableCorFrames'] 

            # Calculate the gain (G) and offset (A) values for the start and end of the calibration
            Gstart = (V1 - V2) / (B1 - B2)
            Astart = V2 - Gstart * B2
        
            Gend = (V4 - V5) / (B4 - B5)
            Aend = V5 - Gend * B5

            # Average all frames for calibration views:
            Astart = np.mean(Astart, axis=0)
            Aend = np.mean(Aend, axis=0)
            Gstart = np.mean(Gstart, axis=0)
            Gend = np.mean(Gend, axis=0)

        else: 
            # Expand radiance from temperuatre calculation: 
            # channel_map = self.load_channel_map(self) --> Not sure why this doesn't work? 
            mat_data = sio.loadmat('./calibration-tables/RadCalIndex.mat')
            channel_map = mat_data['RadCal_Index']['ChannelMap'][0,0].T

            b1_radframe = np.zeros((288, 384))
            b2_radframe = np.zeros((288, 384))
            
            for i in range(15):
                ChannelMask  = channel_map[i,:,:] > 0.5
                b1_radframe[ChannelMask] = B1[i]
                b2_radframe[ChannelMask] = B2[i]

            # Overwrite: 
            B1 = b1_radframe
            B2 = b2_radframe

            # Get the row corrected frames for each view (V1, V2) - Calibration campaign data only has 1 int & bb view:
            V1 = 2**14 - self.row_corrected_data['Int']['bb_look11']['stableCorFrames']
            V2 = 2**14 - self.row_corrected_data['Cold']['cs_look11']['stableCorFrames']

            G_t = (V1 - V2) / (B1 - B2)
            A_t = V2 - G_t * B2
    
        # Initialize a dictionary to store the calibrated radiance for each subgroup
        self.calibrated_products = {}
    
        # Loop through the data for each key and subgroup (Int, Cold, Hot)
        for key in ['Int', 'Cold', 'Hot']:
            for subgroup in getattr(self, key):

                # Print shape of the dataset before processing
                data_shape = self.row_corrected_data[key][subgroup]['stableCorFrames'].shape
                print(f"Processing {key} - {subgroup}: shape {data_shape}")

                # Iterate over frames (first dimension of data_shape)
                num_frames = data_shape[0]

                if not calibration_flag:
                    calibratedRadiance_array = [] 
                    for frame_idx in range(num_frames):
                        # Get the time for the specific observation
                        t = self.times[key][subgroup]

                        # Calculate the current A(t) and G(t) based on the time `t`
                        A_t = Astart + (Aend - Astart) * (t) / (t5 - t1)
                        G_t = Gstart + (Gend - Gstart) * (t) / (t5 - t1)
        
                        # Check if calibration data has the same shape as the scene "Hot" data:
                        if A_t.shape != self.row_corrected_data[key][subgroup]['stableCorFrames'][frame_idx].shape:
                            raise ValueError(f"Shape mismatch: A_t: {A_t.shape}, G_t: {G_t.shape}, Frame: {self.row_corrected_data[key][subgroup]['stableRowSLCorFrames'][frame_idx].shape}")

                        else: 
                            # Calculate the calibrated radiance (without stray light correction)
                            corr_target_rad = (self.row_corrected_data[key][subgroup]['stableCorFrames'][frame_idx] - A_t) / G_t

                        # Store calibrated radiance 
                        calibratedRadiance_array.append(corr_target_rad)

                else: 
                    calibratedRadiance_array = ((2**14 - self.row_corrected_data[key][subgroup]['stableCorFrames']) - A_t) / G_t

                # Store the calibrated radiance in the calibrated_products dictionary
                if key not in self.calibrated_products:
                    self.calibrated_products[key] = {}
    
                if subgroup not in self.calibrated_products[key]:
                    self.calibrated_products[key][subgroup] = {}
                
                self.calibrated_products[key][subgroup]['calibratedRadiance'] = calibratedRadiance_array

                # If key == 'Hot' plot the gain and offset tables:
                if key == 'Hot' or key == 'observation':
                    # Take the mean of all frames if 3D array:
                    if G_t.ndim == 3: 
                        gain = np.mean(G_t, axis=0)
                        offset = np.mean(A_t, axis=0)
                    else: 
                        gain = G_t 
                        offset = A_t

                    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=100) 
                    
                    im0 = axes[0].imshow(offset, aspect='auto', cmap='plasma')
                    axes[0].set_title("Calibration Offset - A(t)")
                    fig.colorbar(im0, ax=axes[0])
                    
                    im1 = axes[1].imshow(gain, aspect='auto', cmap='plasma')
                    axes[1].set_title("Calibraion Gain - G(t)")
                    fig.colorbar(im1, ax=axes[1])
                    
                    plt.tight_layout()
                    plt.show()

    def decimate_tdi_frame(self, frame_data):
        from matplotlib import colormaps

        # Load calibration data: 
        mat_data = sio.loadmat('./calibration-tables/RadCalIndex.mat')
        temperature_gain = mat_data['RadCal_Index']['c2'][0,0]
        channel_map = mat_data['RadCal_Index']['ChannelMapAll'][0,0]
        self.assert_array_shape(temperature_gain, calibration_file=True)   
        self.assert_array_shape(channel_map, calibration_file=True)   

        # Compute the gradient along the columns to detect transitions between channels and take abs to get clear edges
        gradient = np.gradient(channel_map, axis=1)  
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
        summed_data = np.zeros((frame_data.shape[0], len(channel_widths)))
        
        for i, (start_col, end_col) in enumerate(channel_widths):
            summed_data[:, i] = np.sum(frame_data[:, start_col:end_col], axis=1)

        # To match LTM GDS pipeline outputs transpose axis and add 1 at the end: (ASK HENRY WHY)???
        summed_data = np.vstack([summed_data, np.ones((1, summed_data.shape[1]))])
        summed_data = summed_data.T
        
        plt.figure(figsize=(6, 4), dpi=100)
        im3 = plt.imshow(summed_data, aspect='auto', cmap='plasma', interpolation=None)
        plt.title("TDI Adjusted Summed Data") 
        plt.colorbar(im3)
        plt.show()

        return summed_data

    def plot_calibrated_radiance(self, iframe, plot_log=False):
        """
        Plot the calibrated radiance for each key in a new row and each subgroup as columns using a colormap.
        Each row represents one of the keys ('Int', 'Cold', 'Hot'), and each column represents 
        a subgroup within the corresponding key.
        """
        # Get the keys from the calibrated_products dictionary
        keys = ['Int', 'Cold', 'Hot']
    
        # Create a figure with rows equal to the number of keys
        # Int and Cold will have 2 subgroups, Hot will have 1 subgroup
        fig, axes = plt.subplots(nrows=len(keys), ncols=2, figsize=(12, 8))
    
        # Ensure axes is a 2D array even if Hot has only one subgroup
        if len(keys) == 1:
            axes = np.expand_dims(axes, axis=0)
        
        for i, key in enumerate(keys):
            subgroups = list(self.calibrated_products[key].keys())  # Get subgroups for the current key
            
            # Handle the case for 'Hot' which only has 1 subgroup
            if key == 'Hot' and len(subgroups) == 1:
                axes[i, 1].axis('off')  # Disable the second column for 'Hot' since it only has 1 subgroup
    
            for j, subgroup in enumerate(subgroups):
                # Get the calibrated radiance data for the current key and subgroup
                try:
                    calibrated_data = self.calibrated_products[key][subgroup]['calibratedRadiance'][iframe,:,:]
                    ax = axes[i, j]  # Get the current axes
                    
                    if plot_log: 
                        im_cor = ax.imshow(np.log(np.abs(calibrated_data)),  
                                            cmap='plasma', aspect='auto', interpolation="None") 
                    else:
                        im_cor = ax.imshow(calibrated_data, 
                                           vmin=0, vmax=np.max(calibrated_data), cmap='plasma', aspect='auto', interpolation="None")
                        
                        print(np.min((calibrated_data)))
                        print(np.max((calibrated_data)))
                        
                    # Set titles, labels, and other plot settings
                    ax.set_title(f'{key} - {subgroup}')
                    ax.grid(False)  # Disable grid for better visibility with images
    
                    # Add a colorbar to the subplot
                    fig.colorbar(im_cor, ax=ax, orientation='vertical')
    
                except KeyError:
                    # In case there's no data for a specific subgroup (e.g., missing subgroups)
                    ax = axes[i, j]
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                    ax.set_title(f'{key} - {subgroup}')
                    ax.axis('off')
    
        # Adjust layout to avoid overlap
        plt.tight_layout()
        plt.show()

    def plot_corrected_data(self, group_name, iframe):
        """
        This function will plot the raw data, corrected data, and their difference for each subgroup in a given group.
        It will display them in a grid format, with rows for subgroups and columns for raw and corrected data.
        
        :param group_name: The name of the group to plot ('Int', 'Cold', or 'Hot')
        :param iframe: The index of the frame to plot (for example, if you want to plot a specific frame)
        """
        # Ensure that the group exists in the RadCal object
        if group_name not in self.row_corrected_data:
            print(f"Error: Group '{group_name}' not found in the RadCal object.")
            return
        
        # Get the subgroups for this group
        subgroups = self.row_corrected_data[group_name]
        
        # Number of subgroups
        num_subgroups = len(subgroups)
        
        # Create the figure and axes for the grid plot
        fig, axes = plt.subplots(nrows=num_subgroups, ncols=5, figsize=(20, num_subgroups * 4), dpi=100)
        
        # Ensure axes is iterable even if there's only one row
        if num_subgroups == 1:
            axes = [axes]
        
        # Define titles for each subplot column
        titles = ['Raw Data', 'Corrected Frames', 'Stable Corrected Frames', 'Difference']
    
        # Iterate over subgroups and plot each data type (raw, corrected, stable, difference)
        for idx, (subgroup, row_corrected_data) in enumerate(subgroups.items()):
            raw_data = row_corrected_data.get('rawdata')
            cor_frames = row_corrected_data.get('corFrames')
            stable_cor_frames = row_corrected_data.get('stableCorFrames')
            straylight_cor_frames = row_corrected_data.get('stableRowSLCorFrames')
    
            # Define the data for each subplot
            data = {
                'Raw Data': raw_data[iframe,:,:],
                'Corrected Frames': cor_frames[iframe,:,:],
                'Stable Corrected Frames': stable_cor_frames[iframe,:,:],
                'Stray Light Corrected Frames': straylight_cor_frames[iframe,:,:],
                'Difference': raw_data[iframe,:,:] - straylight_cor_frames[iframe,:,:]
            }
            # Calculate vmin and vmax for the first three plots (raw, corrected, stable) based on log-transformed data
            # all_data_values = np.concatenate([
            #     data['Raw Data'].ravel(),
            #     data['Corrected Frames'].ravel(),
            #     data['Stable Corrected Frames'].ravel()
            # ])
            # vmin, vmax = np.min(all_data_values), np.max(all_data_values)
    
            # Plot each subplot and calculate individual vmin and vmax
            for col_idx, (title, plot_data) in enumerate(data.items()):
                im = axes[idx][col_idx].imshow(plot_data, cmap='plasma' if title != 'Difference' else 'bwr', aspect='auto', interpolation="None")
                axes[idx][col_idx].set_title(f"{subgroup} {title}")
    
                # Calculate vmin and vmax independently based on the plot data
                if title != 'Difference':
                    # im.set_clim(vmin=vmin, vmax=vmax)
                    plot_vmin, plot_vmax = plot_data.min(), plot_data.max()
                    im.set_clim(vmin=plot_vmin, vmax=plot_vmax)
                    
                else:
                    # For difference, use symmetric scaling
                    diff_vmax = np.max(np.abs(plot_data))
                    diff_vmin = -diff_vmax
                    im.set_clim(vmin=diff_vmin, vmax=diff_vmax)
    
                # Add colorbars
                fig.colorbar(im, ax=axes[idx][col_idx], orientation='vertical')
    
        # Adjust layout for better presentation
        plt.tight_layout()
        plt.show()
