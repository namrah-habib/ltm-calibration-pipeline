from dataclasses import dataclass, field
import h5py    
import numpy as np    
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.ltm_model import LTM_Model
from src.preview_frame import preview_frame

@dataclass
class RadCal:
    """Class to handle loading, storing & processing LTM data."""

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
        """Initialize empty nested dictionaries used by RadCal.

        Parameters:
            None

        Raises:
            None

        Updates RadCal:
            row_corrected_data:         Sets empty dicts for 'Int', 'Cold', 'Hot'.
            times/tdi_flg/temps:        Sets empty dicts for 'Int', 'Cold', 'Hot'.
        """

        self.row_corrected_data = {k: {} for k in ['Int', 'Cold', 'Hot']}
        self.times = {k: {} for k in ['Int', 'Cold', 'Hot']}
        self.tdi_flg = {k: {} for k in ['Int', 'Cold', 'Hot']}
        self.temps = {k: {} for k in ['Int', 'Cold', 'Hot']} 

    def assert_array_shape(self, array, calibration_file=False):
        """Validate expected array dimensionality for LTM data.

        Parameters:
            array (np.ndarray):         Input data array to validate.
            calibration_file (bool):    True if data is a calibration table (2-D expected).

        Raises:
            TypeError:                  If array is not a NumPy array.
            ValueError:                 If dimensionality or sizes do not match expectations.

        Updates RadCal:
            None
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

    def load_data(self, file_path: str, calibration_flag: bool = False):
        """Load the LTM HDF5 raw data and extract data and metadata into RadCal object.

        Parameters:
            file_path (str):            Path to the HDF5 file.
            calibration_flag (bool):    True if file is from the LTM FM calibration campaign.

        Raises:
            FileNotFoundError:          If the file cannot be opened.
            KeyError:                   If expected groups/datasets are missing.
            ValueError:                 If array shapes do not match expectations.

        Updates RadCal:
            Int/Cold/Hot:               Raw arrays per look.
            times/tdi_flg/temps:        Per-look metadata (when present).
        """
        
        with h5py.File(file_path, 'r') as f:
            if calibration_flag:
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
                        
                # Extract times & temperatures:
                for category, subgroups in mapping.items():
                    if category in f:
                        for subgroup_name, dataset_name in subgroups.items():
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
            print(f"Data successfully loaded from {file_path}")
    
    def load_temperature_gain_table(self, tdi_flag: bool = True):
        """Load the temperature-gain look-up table for detector correction.

        Parameters:
            tdi_flag (bool):            True to load TDI-specific table; False loads non-TDI table.

        Raises:
            FileNotFoundError:          If the required pickle is missing.
            ValueError:                 If the table shape is invalid.

        Updates RadCal:
            None

        Returns:
            np.ndarray:                 2-D gain-ratio table used in row correction.
        """            
        if tdi_flag:
            with open('./calibration-tables/TemperatureGain_tdi.pkl', 'rb') as f:
                b2 = pickle.load(f)
    
        else:
            with open('./calibration-tables/TemperatureGain.pkl', 'rb') as f:
                b2 = pickle.load(f).T
                
        self.assert_array_shape(b2, calibration_file=True)      

        return b2

    def load_channel_map(self):
        """Load the channel map from RadCalIndex.mat.

        Parameters:
            None

        Raises:
            FileNotFoundError:          If RadCalIndex.mat is missing.
            ValueError:                 If the channel map shape is invalid.

        Updates RadCal:
            None

        Returns:
            np.ndarray:                 2-D channel map (288×384/385 expected).
        """
        mat = sio.loadmat('./calibration-tables/RadCalIndex.mat')
        channel_map = mat['RadCal_Index']['ChannelMap'][0,0].T
        if not isinstance(channel_map, np.ndarray) or channel_map.ndim != 3 or channel_map.shape[0] != 15:
            raise ValueError(f"ChannelMap expected shape (15, rows, cols); got {None if not hasattr(channel_map,'shape') else channel_map.shape}")
        return channel_map

    def load_bad_pixel_table(self):
        """Load (or compute) a bad-pixel table.

        Parameters:
            None

        Raises:
            NotImplementedError:        If not yet implemented.

        Updates RadCal:
            None

        Returns:
            None
        """
        print("Still needs to be added")
        return

    def apply_row_correction(self, mask_ind: int = 365):
        """Apply row & offset correction (and stray-light correction) to all looks.

        Parameters:
            mask_ind (int):             Column index where masked region begins (used for offsets).

        Raises:
            FileNotFoundError:          If required calibration tables are missing.
            ValueError:                 If shapes are inconsistent.

        Updates RadCal:
            row_corrected_data[key][subgroup]:
                • 'rawdata'                     : original frames (copy)
                • 'offsetAverage'               : per-row offsets
                • 'corFrames'                   : row-corrected frames
                • 'stableCorFrames'             : stabilized temperature-gain row-corrected frames
                • 'stableRowSLCorFrames'        : stabilized frames with stray light removed
        """
        for key in ['Int', 'Cold', 'Hot']:
            for subgroup in getattr(self, key):
                b2 = self.load_temperature_gain_table(tdi_flag=self.tdi_flg[key][subgroup])
            
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

                # Get the avg value over all masked pixels during internal calibration at the start of the observation: 
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
        """Apply per-row offset removal (and optional gain-ratio scaling).

        Parameters:
            M (np.ndarray):             2-D frame (rows×cols) to correct.
            mask_ind (int):             Column index where masked region begins.
            avg_int_offset (float):     Global internal offset of masked pixels.
            gain_ratio (np.ndarray):    Optional 2-D array matching M for stabilization.

        Raises:
            ValueError:                 If gain_ratio is provided but incompatible.

        Updates RadCal:
            None

        Returns:
            np.ndarray:                 Corrected frame (same shape as M).
        """
        # If gain_ratio is not provided, create a default one (array of ones with the same shape as M)
        if gain_ratio is None:
            gain_ratio = np.ones_like(M)

        # Ensure gain_ratio has the same shape as M
        if gain_ratio.shape[1] != M.shape[1]:
            raise ValueError("gain_ratio must have the same shape as M")
        
        return M - gain_ratio * (np.mean(M[:, mask_ind:-1], axis=1) - avg_int_offset)[:, np.newaxis]

    def row_cor_offset(self, M: np.ndarray, mask_ind: int) -> np.ndarray:
        """Compute the average masked-column value per row.

        Parameters:
            M (np.ndarray):             2-D frame (rows×cols).
            mask_ind (int):             Column index where masked region begins.

        Raises:
            None

        Updates RadCal:
            None

        Returns:
            np.ndarray:                 1-D per-row average over masked pixels.
        """
        
        return np.mean(M[:, mask_ind:-1], axis=1)

    def stray_light_correction(self, M: np.ndarray, key, subgroup):
        """Subtract the measured stray-light (from FM ground calibration campaign) pattern from a frame.

        Parameters:
            M (np.ndarray):             2-D stabilized row-corrected frame.
            key (str):                  One of 'Int', 'Cold', 'Hot'.
            subgroup (str):             Look name within the key.

        Raises:
            FileNotFoundError:          If stray light table is missing.
            ValueError:                 If table shape is invalid.

        Updates RadCal:
            None

        Returns:
            np.ndarray:                 Frame after stray-light subtraction.
        """
        if self.tdi_flg[key][subgroup]:
            with open('./calibration-tables/StrayLightCorr_tdi.pkl', 'rb') as f:
                measured_straylight = pickle.load(f)
        else:
            with open('./calibration-tables/StrayLightCorr.pkl', 'rb') as f:
                measured_straylight = pickle.load(f).T
                
        self.assert_array_shape(measured_straylight, calibration_file=True)   

        return M - measured_straylight

    def calc_calibrated_radiance(self, calibration_flag=False, plot_gain_offset: bool = False):
        """Compute calibrated radiance using a 4-point calibration for each LTM observation.

        Parameters:
            calibration_flag (bool):    True for calibration-campaign files (2 looks only = 2-point calibration).
            plot_gain_offset (bool):    If True, plot G(t) and A(t) diagnostics.

        Raises:
            TypeError:                  If required calibration views are missing (obs path).
            ValueError:                 If shapes are inconsistent.

        Updates RadCal:
            - calibrated_products[key][subgroup]['calibratedRadiance']: list/array of frames.
        """
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
        # and the frames I comparing to have already removed stray light contribution. 
        model = LTM_Model()
        B1 = model.calcRadiance(model.channelFilterMap, [T1]) 
        B2 = model.calcRadiance(model.channelFilterMap, [T2]) 
        B4 = model.calcRadiance(model.channelFilterMap, [T4]) 
        B5 = model.calcRadiance(model.channelFilterMap, [T5])

        # Check if you have calibration veiws before & after data (but code should have failed by now if you didn't..)
        if not calibration_flag:
            missing = []
            for g,k in [('Int','bb_look11'),('Int','bb_look21'),
                        ('Cold','cs_look11'),('Cold','cs_look21')]:
                if k not in self.temps.get(g, {}):
                    missing.append(f"{g}/{k}")
            if missing:
                raise KeyError(f"Missing calibration views: {', '.join(missing)}")
            
            # Get the corrected frames for each view (V1, V2, V4, V5)
            V1 = self.row_corrected_data['Int']['bb_look11']['stableRowSLCorFrames']
            V2 = self.row_corrected_data['Cold']['cs_look11']['stableRowSLCorFrames']
            V4 = self.row_corrected_data['Int']['bb_look21']['stableRowSLCorFrames']
            V5 = self.row_corrected_data['Cold']['cs_look21']['stableRowSLCorFrames'] 

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
            channel_map = self.load_channel_map()  

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
            V1 = 2**14 - self.row_corrected_data['Int']['bb_look11']['stableRowSLCorFrames']
            V2 = 2**14 - self.row_corrected_data['Cold']['cs_look11']['stableRowSLCorFrames']

            G_t = (V1 - V2) / (B1 - B2)
            A_t = V2 - G_t * B2
    
        # Initialize a dictionary to store the calibrated radiance for each subgroup
        self.calibrated_products = {}
    
        # Loop through the data for each key and subgroup (Int, Cold, Hot)
        for key in ['Int', 'Cold', 'Hot']:
            for subgroup in getattr(self, key):
                data_shape = self.row_corrected_data[key][subgroup]['stableRowSLCorFrames'].shape
               
                # Iterate over frames (first dimension of data_shape)
                num_frames = data_shape[0]

                if not calibration_flag:
                    calibratedRadiance_array = [] 
                    for frame_idx in range(num_frames):
                        # Get the time for the specific observation
                        t = self.times[key][subgroup]

                        # Calculate the current A(t) and G(t) based on the time `t`
                        den = (t5 - t1)
                        if den == 0: den = 1.0
                        alpha = (t - t1) / den
    
                        A_t = Astart + alpha * (Aend - Astart)
                        G_t = Gstart + alpha * (Gend - Gstart)
        
                        corr_target_rad = (self.row_corrected_data[key][subgroup]['stableRowSLCorFrames'][frame_idx] - A_t) / G_t

                        # Store calibrated radiance 
                        calibratedRadiance_array.append(corr_target_rad)

                else: 
                    calibratedRadiance_array = ((2**14 - self.row_corrected_data[key][subgroup]['stableRowSLCorFrames']) - A_t) / G_t

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

                    if plot_gain_offset: 
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
        """Decimate a full-resolution frame to TDI channels via channel boundaries.

        Parameters:
            frame_data (np.ndarray):    2-D frame (rows×cols) to be decimated.

        Raises:
            FileNotFoundError:          If RadCalIndex.mat is missing.
            ValueError:                 If loaded calibration arrays are invalid.

        Updates RadCal:
            None

        Returns:
            np.ndarray:                 2-D TDI-summed data (channels×rows) after reformat.
        """
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
        """Plot calibrated radiance grids across keys/subgroups.

        Parameters:
            iframe (int):               Frame index to visualize.
            plot_log (bool):            If True, plot log(|radiance|).

        Raises:
            KeyError:                   If expected keys/subgroups are missing.

        Updates RadCal:
            None

        Returns:
            None
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
                                            cmap='plasma', aspect='auto',interpolation=None) 
                    else:
                        im_cor = ax.imshow(calibrated_data, 
                                           vmin=0, vmax=np.max(calibrated_data), cmap='plasma', aspect='auto',interpolation=None)
                        
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
        """Plot raw, corrected, stabilized, stray-light-corrected, and difference frames.

        Parameters:
            group_name (str):           'Int', 'Cold', or 'Hot'.
            iframe (int):               Frame index to plot.

        Raises:
            KeyError:                   If the group/subgroup is not present.

        Updates RadCal:
            None

        Returns:
            None
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
    
            # Plot each subplot and calculate individual vmin and vmax
            for col_idx, (title, plot_data) in enumerate(data.items()):
                im = axes[idx][col_idx].imshow(plot_data, cmap='plasma' if title != 'Difference' else 'bwr', aspect='auto',interpolation=None)
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

    def save_calibrated_data(self, out_path, overwrite=False):
        """Write a flat, per-look HDF5 with dataset names.

        Parameters:
            out_path (str|Path):        Destination .h5 path.
            overwrite (bool):           If False and file exists → skip (print and return).

        Raises:
            None (function prints and returns if file exists and overwrite=False)

        Updates RadCal:
            None (writes to disk only)

        Returns:
            None
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not overwrite:
            print(f"  -> output file exists: {out_path}  — skipping")
            return

        def _to_np(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return x
            try:
                return np.array(x)
            except Exception:
                return None

        def _unique_name(g, base):
            """Avoid collisions if something with the same name already exists."""
            name = base
            k = 1
            while name in g:
                name = f"{base} ({k})"
                k += 1
            return name

        def _write_flat(g, base, value):
            """ Write value as one or more datasets into group g using 'base' as name. """
            if value is None:
                return
            if isinstance(value, list):
                arr = _to_np(value)
                if arr is not None and arr.dtype != object:
                    g.create_dataset(_unique_name(g, base), data=arr)
                else:
                    for i, v in enumerate(value):
                        if v is None:
                            continue
                        g.create_dataset(_unique_name(g, f"{base} {i:04d}"), data=_to_np(v))
                return
            g.create_dataset(_unique_name(g, base), data=_to_np(value))

        def _set_attr(g, key, val):
            """Always store as an attribute (supports scalars or numpy arrays)."""
            if val is None:
                return
            arr = _to_np(val)
            if arr is None:
                return
            # h5py supports numeric/boolean/string and numpy arrays as attribute values
            g.attrs[key] = arr

        # Role labels (change if you prefer 'hot/cold/scene')
        role_map = {
            'Int':  'Internal Calibration View',
            'Cold': 'Cold Calibration View',
            'Hot':  'Scene',
        }

        # Optional structures on the object
        rc    = getattr(self, "row_corrected_data", {}) or {}
        cal   = getattr(self, "calibrated_products", {}) or {}
        times = getattr(self, "times", {}) or {}
        tdi   = getattr(self, "tdi_flg", {}) or {}
        temps = getattr(self, "temps", {}) or {}

        # Name mappings for row-corrected & calibrated keys
        rc_name_map = {
            "rawdata": None,                        # skip writing rawdata again
            "corFrames": "Row Corrected Frames",
            "offsetAverage": "Offset Average",
            "stableCorFrames": "Corrected Frames",
            "stableRowSLCorFrames": "Corrected Frames with Stray Light Removed",
        }
        cal_name_map = {
            "calibratedRadiance": "Calibrated Radiance",
        }

        with h5py.File(out_path, "w") as f:
            # walk the three source dicts
            for src_name in ("Int", "Cold", "Hot"):
                src_dict = getattr(self, src_name, None)
                if not isinstance(src_dict, dict) or not src_dict:
                    continue

                role = role_map.get(src_name, src_name)
                role_grp = f.require_group(role)

                for look_name, raw_arr in src_dict.items():
                    look_grp = role_grp.create_group(str(look_name))

                    # --- Raw frames ---
                    _write_flat(look_grp, "Raw Data", raw_arr)  # renamed from "raw"

                    # --- Row-corrected content (flattened, renamed) ---
                    rc_map = rc.get(src_name, {}).get(look_name, {})
                    if isinstance(rc_map, dict) and rc_map:
                        for k, v in rc_map.items():
                            pretty = rc_name_map.get(k, None)
                            if pretty is None:
                                # skip if explicitly None (e.g., rawdata)
                                continue
                            # Unknown keys: keep original name (or make a friendly fallback)
                            name_to_use = pretty if pretty else k
                            _write_flat(look_grp, name_to_use, v)

                    # --- Calibrated/products content (flattened, renamed) ---
                    cal_map = cal.get(src_name, {}).get(look_name, {})
                    if isinstance(cal_map, dict) and cal_map:
                        for k, v in cal_map.items():
                            pretty = cal_name_map.get(k, None)
                            name_to_use = pretty if pretty else k  # only rename known key(s)
                            _write_flat(look_grp, name_to_use, v)

                    # --- Per-look attributes (always attributes) ---
                    _set_attr(look_grp, "time",        times.get(src_name, {}).get(look_name))
                    _set_attr(look_grp, "temperature", temps.get(src_name, {}).get(look_name))  # renamed from "temp"
                    _set_attr(look_grp, "tdi_flg",     tdi.get(src_name, {}).get(look_name))

        print(f"Saved RadCal data to {out_path}")

    def plot_quicklook(self, quicklook_path):
        """Render and save a quicklook of the time-averaged calibrated scene radiance.

        Parameters:
            quicklook_path (str | pathlib.Path):  Output path (e.g., PNG/PDF) for the saved quicklook figure.

        Raises:
            KeyError:       If 'Hot' → 'scene' → 'calibratedRadiance' is missing from self.calibrated_products.
            ValueError:     If the calibrated radiance array is empty or not 3-D (frames × rows × cols).
            OSError:        If the quicklook image cannot be written to the given path.

        Updates RadCal:
            None. This function is read-only; it computes a frame-average and writes a figure to disk.
        """
        cal_map = self.calibrated_products['Hot']['scene']['calibratedRadiance']
        avg_frame = np.nanmean(cal_map, axis=0)
        frame_avg = np.nanmean(avg_frame)
        frame_std = np.nanstd(avg_frame)

        preview_frame(
            avg_frame,
            clim=(0, frame_avg + 1.5 * frame_std),
            title=f"Calibrated Average Scene Radiance",
            ctitle="Radiance",                   
            plot_histogram=False,
            plot_colorbar=True,
            save_file=quicklook_path,
        )

    def describe(self):
        """Print a human-readable tree of the RadCal object (data, products, and metadata).

        Parameters:
            None.

        Raises:
            None.

        Updates RadCal:
            None. This function only inspects and prints the in-memory structure.
        """

        def _summ(x):
            if x is None:
                return "None"
            if isinstance(x, np.ndarray):
                return f"ndarray{tuple(x.shape)} {x.dtype}"
            if isinstance(x, (int, float, bool, str)):
                return f"{type(x).__name__}={x}"
            if isinstance(x, dict):
                return f"dict[{len(x)}]"
            if isinstance(x, (list, tuple)):
                return f"{type(x).__name__}[{len(x)}]"
            return type(x).__name__

        def _walk_dict(d, indent="  "):
            for k, v in d.items():
                if isinstance(v, dict):
                    print(f"{indent}• {k}/  ({_summ(v)})")
                    _walk_dict(v, indent + "  ")
                else:
                    print(f"{indent}• {k}  ({_summ(v)})")

        print("RadCal structure")
        print("────────────────")
        for name in ("Int", "Cold", "Hot"):
            val = getattr(self, name, None)
            print(f"{name}/  ({_summ(val)})")
            if isinstance(val, dict):
                _walk_dict(val)

        for name in ("row_corrected_data", "calibrated_products"):
            val = getattr(self, name, {})
            print(f"{name}/  ({_summ(val)})")
            if isinstance(val, dict):
                _walk_dict(val)

        # Key metadata
        print("metadata/")
        for name in ("times", "tdi_flg", "temps"):
            val = getattr(self, name, {})
            print(f"  • {name}/ ({_summ(val)})")
            if isinstance(val, dict):
                _walk_dict(val, indent="    ")
