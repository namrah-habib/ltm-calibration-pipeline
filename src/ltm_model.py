import numpy as np
import matplotlib.pyplot as plt
import pickle

class LTM_Model:
    def __init__(self, frameRate=50, VID_PW=350):
        # Observation specific parameters
        self.Integration_Time = 0.033     # seconds
        self.Temperature_Scene = 100      # Kelvin
        self.TDIaverage = 9  

        # Constant optical + detector parameters
        self.fDet = frameRate            #  Hz
        self.VID_PW = VID_PW             # mW
        self.intTime = 1 / self.fDet  
        self.NEP = 30e-12  
        self.pixelpitch = 35e-6  
        self.pixelarea = self.pixelpitch ** 2  
        self.FNumberDetector = 1.5 
        self.OmegaDetector = 2 * np.pi * (1 - np.cos(np.arctan(1 / (2 * self.FNumberDetector))))  
        self.aOmega = self.pixelarea * self.OmegaDetector  

        # Load data from MAT files
        self.wavelengths, self.channelTransmission = self.load_filter_data()  
        self.Mirror_Reflectivity = self.load_mirror_data()  
        self.DetectorResponse = self.load_detector_response()  

        # Channel specific parameters
        self.channelFilterMap = list(reversed([15, 13, 11, 9, 7, 5, 3, 12, 1, 2, 4, 6, 8, 10, 15]))  
        self.channelName = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'B1', 'B2']  
        self.channelCenter = [7, 7.25, 7.5, 7.8, 8, 8.28, 8.55, 8.75, 9, 9.375, 9.5, 10, 18.75, 37.5, 75]  
        self.channelWidth = [0.25] * 11 + [6.5, 12.5, 25, 50]  

        # Simulated temperatures and calculations
        self.SimTemperatures = np.arange(50, 405, 5)  
        # self.calculate_radiance_and_noise()  

    # Method to load filter data from a .mat file
    def load_filter_data(self):
        with open('./calibration-tables/LTM_MODEL_FilterBandpasses.pkl', 'rb') as f:
            data = pickle.load(f)
            
        wavelengths = data['FilterBandpasses'][:, 0]
        channelTransmission = [data['FilterBandpasses'][:, i+1] / 100 for i in range(15)]
        
        for i in range(13, 15):
            channelTransmission[i] *= 0.66  # Apply scaling for quartz window effect
            
        return wavelengths, channelTransmission

    # Method to load mirror reflectivity data
    def load_mirror_data(self):
        with open('./calibration-tables/LTM_MODEL_MirrorReflectivity.pkl', 'rb') as f:
            data = pickle.load(f)
            
        return data['MirrorReflectivity']

    # Method to load detector response data
    def load_detector_response(self):
        with open('./calibration-tables/LTM_MODEL_DetectorResponse.pkl', 'rb') as f:
            data = pickle.load(f)

        return data['InoDetectorResponse'].T

    # Main calculation method for radiance, power, SNR, and NETD
    def calculate_radiance_and_noise(self):
        # Calculate ideal and actual radiance, power, SNR, and NETD (Noise Equivalent Temperature Difference)
        self.IdealRadianceData = self.IdealRadiance(np.arange(1, 16), self.SimTemperatures)
        self.IdealPowerData = self.IdealRadianceData * self.aOmega  # Power for ideal conditions
        self.IdealSNRData = self.IdealPowerData / self.NEP  # Signal-to-noise ratio for ideal conditions

        self.RadianceData = self.calcRadiance(np.arange(1, 16), self.SimTemperatures)
        self.PowerData = self.RadianceData * self.aOmega
        self.SNRData = self.PowerData / self.NEP

        # Calculate NETD by differentiating power with respect to temperature
        dPdT = (self.calcRadiance(np.arange(1, 16), self.SimTemperatures + 0.01) -
                self.calcRadiance(np.arange(1, 16), self.SimTemperatures)) / 0.01 * self.aOmega
        self.NETDData = self.NEP / dPdT  # NETD calculation

    def calcRadiance(self, channelnum, Temp):
        wav = self.wavelengths * 1e-6                           #  meters
        OpticsTotalReflectance = self.Mirror_Reflectivity ** 6  # Total reflectance after multiple reflections
        Radiance = np.zeros((len(channelnum), len(Temp)))

        for i, channel in enumerate(channelnum):
            for j, temp in enumerate(Temp):
                planck_spectrum = self.planck(wav, temp)
                transmission = self.channelTransmission[channel - 1]  
                detector_response = self.DetectorResponse[:,0]
                reflectance = OpticsTotalReflectance[:,0]
    
                # Ensure all terms are aligned for element-wise multiplication
                product = planck_spectrum * reflectance * detector_response * transmission
    
                # Integrate the spectral radiance across the filter bandpass for each channel and temperature
                Radiance[i, j] = np.trapz(product, wav)  
    
        return Radiance

    # Ideal radiance calculation for simulated transmission response
    def IdealRadiance(self, channelnum, Temp):
        wav = self.wavelengths * 1e-6
        Radiance = np.zeros((len(Temp), len(channelnum)))

        for i, channel in enumerate(channelnum):
            T = np.ones(len(wav))  # Transmission mask initialization
            T[self.wavelengths < self.channelCenter[channel - 1] - self.channelWidth[channel - 1]] = 0
            T[self.wavelengths > self.channelCenter[channel - 1] + self.channelWidth[channel - 1]] = 0
            for j, temp in enumerate(Temp):
                Radiance[j, i] = np.trapz(self.planck(wav, temp) * T, wav)
                
        return Radiance

    # blackbody radiation at a given wavelength and temperature
    def planck(self, wavelength, temp):
        c1 = 1.191042e-16  
        c2 = 0.014387752  
        spectrum = c1 / (wavelength ** 5 * (np.exp(c2 / (wavelength * temp)) - 1))
        
        return spectrum

    # Plot methods for NETD, power, SNR, radiance, and transmissions
    def plotNETD(self):
        plt.figure()
        for i in range(15):
            plt.semilogy(self.SimTemperatures, self.NETDData[i, :], label=self.channelName[i])
        plt.grid(True)
        plt.ylabel('NETD / K')
        plt.ylim([1e-2, 2e3])
        plt.xlabel('Temperature')
        plt.legend(loc='best', ncol=2)
        plt.title('LTM Model NETD - using measured NEP')
        plt.show()

    def plotPower(self):
        plt.figure()
        for i in range(15):
            plt.semilogy(self.SimTemperatures, self.PowerData[i, :], label=self.channelName[i])
        plt.grid(True)
        plt.ylabel('Power W')
        plt.xlabel('Temperature')
        plt.legend(loc='best', ncol=2)
        plt.show()

    def plotSNR(self, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.gca()

        SNRData = self.IdealSNRData if self.idealFlag else self.SNRData
        for i in range(15):
            ax.semilogy(self.SimTemperatures, SNRData[i, :], label=self.channelName[i])

        ax.grid(True)
        ax.set_ylabel('SNR')
        ax.set_ylim([1, np.max(SNRData)])
        ax.set_xlabel('Temperature')
        ax.legend(loc='best', ncol=2)
        ax.set_title('LTM Model SNR - using measured NEP')
        plt.show()

    def plotcalcRadiances(self, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.gca()

        for i in range(15):
            ax.semilogy(self.SimTemperatures, self.RadianceData[i, :], label=self.channelName[i])

        ax.grid(True)
        ax.set_ylabel('Radiance W/m^2/sr')
        ax.set_ylim([1e-3, 2e3])
        ax.set_xlabel('Temperature')
        ax.legend(loc='best', ncol=2)
        plt.show()

    def plotTransmissions(self, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.gca()

        for i in range(15):
            ax.semilogx(self.wavelengths, self.channelTransmission[i], label=self.channelName[i])

        ax.grid(True)
        ax.set_xlabel('Wavelength /µm')
        ax.set_xlim([5, 160])
        ax.set_xticks([5, 10, 20, 40, 80, 160])
        ax.set_ylabel('Transmission')
        ax.legend(loc='best', ncol=2)
        plt.show()
    