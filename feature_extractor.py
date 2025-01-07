import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
import mne
from mne.decoding import CSP
import pywt
from scipy.stats import entropy
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import euclidean
from scipy import signal
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from fastdtw import fastdtw

# データ読み込みと特徴量抽出のためのカスタムトランスフォーマー
class EEGCSVreader(BaseEstimator, TransformerMixin):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_encoder = LabelEncoder()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        features_list = []
        labels_list = []
        
        for label in ['frontside_kickturn', 'backside_kickturn', 'pumping','reverse_pumping']:
            label_dir = os.path.join(self.data_dir, label)
            for file in os.listdir(label_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(label_dir, file)
                    try:
                        trial_data = pd.read_csv(file_path)
                        if trial_data.shape != (251, 72):
                            print(f"Skipping file {file} due to incorrect shape: {trial_data.shape}")
                            continue
                        # features = self.extract_features(trial_data)
                        # やっぱり251を250にする
                        trial_data = trial_data.iloc[1:,:]
                        features_list.append(trial_data)
                        labels_list.append(label)
                    except Exception as e:
                        print(f"Error processing file {file}: {str(e)}")
        
        if not features_list:
            raise ValueError("No valid data found. Check your data directory and file formats.")
        
        X = features_list
        y = self.label_encoder.fit_transform(labels_list)
        # 元のクラスラベルと対応するエンコードされたラベルを表示


        return X, y
    

class EEGFeatureExtractor():
    '''
    EEGデータから特徴量を抽出するクラス

    Methods
    -------
    extract(train)
        trainデータから特徴量を抽出するメソッド．train全体の統計量を用いて特徴量を抽出する．

    apply(test)    
        testデータから特徴量を抽出するメソッド．trainデータの統計量を用いて特徴量を抽出する．
    '''

    def __init__(self):
        self.extracted = False
        self.feature_names = []
        self.channel_pos = pd.read_csv('channel_pos.csv',index_col=0)
        self.bands = {'delta':[0.5,3.5],'theta':[3.5,7.5],'alpha':[7.5,13.5],'beta':[13.5,30]}

         

    def extract(self, x_train, y_train):
        '''
        trainデータから特徴量を抽出するメソッド．train全体の統計量を用いて特徴量を抽出する．
        params
            x_train : EEG(251x72) for each trial
            y_train : label for each trial
        returns
            features : 抽出した特徴量リスト for each trial

        '''
        # ラベル毎にチャネルの平均波形を算出
        self.mean_eeg = []
        self.std_eeg = []
        self.labels = np.unique(y_train)
        self.channels = x_train[0].columns
        for l in self.labels:
            x_train_l = [x_train[i] for i in range(len(y_train)) if y_train[i] == l]
            mean_eeg_l = pd.DataFrame()
            std_eeg_l = pd.DataFrame()
            for c in self.channels:
                ch_waves = pd.concat([x_train_l[i][c] for i in range(len(x_train_l))], axis=1)
                mean_eeg_l[c] = ch_waves.mean(axis=1)
                std_eeg_l[c] = ch_waves.std(axis=1)
            self.mean_eeg.append(mean_eeg_l) 
            self.std_eeg.append(std_eeg_l)

        # ICA
        info = mne.create_info(ch_names=self.channels.to_list(), sfreq=250, ch_types='eeg') 
        epochs = mne.EpochsArray(np.array(x_train).transpose(0,2,1), info)
        self.ica = mne.preprocessing.ICA(n_components=0.95, random_state=97)
        self.ica.fit(epochs)

        # CSP
        self.CSPs = []
        for l in self.labels:
            csp = CSP(n_components=len(self.channels), transform_into='csp_space')
            csp.fit(np.array(x_train).transpose(0,2,1), np.where(y_train == l, 0, 1))
            self.CSPs.append(csp)


        self.extracted = True
        
        # 各trialの特徴量を設定
        features = self.apply(x_train, add_feature_names=True)
        return features
    
    def apply(self, X, add_feature_names=False):
        '''
        データXから特徴量を抽出するメソッド．trainデータの統計量を用いて特徴量を抽出する．
        この機構はtrainとtestで共通である
        '''
        if not self.extracted:
            raise ValueError("You must extract features from training data before applying to test data.")
        
        features = []
        # for eeg in X:   # eegは1トライアル分のデータ
        for eeg_idx in tqdm(range(len(X))):
            eeg = X[eeg_idx]
            trial_features = []
            reg_split = 3 

            # 特徴量１：各チャネルの平均波形からの距離
            for lidx in range(len(self.labels)):

                # 平均波形からの偏差
                mean_eeg_l = self.mean_eeg[lidx]
                std_eeg_l = self.std_eeg[lidx]

                norm_dev = (eeg - mean_eeg_l) / std_eeg_l
                abs_norm_dev = norm_dev.abs()       # shape = (251, 72)

                # 3分割してからmeanを取る
                dev1, dev2, dev3 = np.array_split(abs_norm_dev, reg_split, axis=0)
                trial_features.extend(dev1.mean().tolist())
                trial_features.extend(dev2.mean().tolist())
                trial_features.extend(dev3.mean().tolist())

            # 特徴量1-2:各チャネルの平均波形からの距離（相関係数）
            for lidx in range(len(self.labels)):
                mean_eeg_l = self.mean_eeg[lidx]
                for eeg_part,mean_part in zip(np.array_split(eeg, reg_split, axis=0), np.array_split(mean_eeg_l, reg_split, axis=0)):
                    eeg_df = pd.DataFrame(eeg_part, columns=self.channels)
                    mean_df = pd.DataFrame(mean_part, columns=self.channels)
                    corr = eeg_df.corrwith(mean_df)
                    trial_features.extend(corr.tolist())

            # 特徴量1-3:各チャネルの平均波形からの距離（DTW距離）
            for lidx in range(len(self.labels)):
                mean_eeg_l = self.mean_eeg[lidx]
                for c in self.channels:
                    eeg_c = eeg[c].to_numpy().reshape(-1,1)
                    mean_c = mean_eeg_l[c].to_numpy().reshape(-1,1)
                    distance, _ = fastdtw(eeg_c, mean_c,dist=euclidean)
                    trial_features.append(distance)

            # 特徴量２：各チャネルの平均波形（分割）の線形回帰の傾きと切片
            for eeg_part in np.array_split(eeg, reg_split, axis=0):
                slope = eeg_part.apply(lambda x: np.polyfit(range(eeg_part.shape[0]), x, 1)[0]).tolist()
                bias = eeg_part.apply(lambda x: np.polyfit(range(eeg_part.shape[0]), x, 1)[1]).tolist()
                trial_features.extend(slope)
                trial_features.extend(bias)

            # 特徴量３：電位重心（電位で重み付け平均をすることで電位の中心を求める）
            # 波形を５分割してから重心を求める
            for eeg_part in np.array_split(eeg, 5, axis=0):
                x, y = 0,0
                for c in self.channels:
                    x += eeg_part[c].mean()*(self.channel_pos.loc['x',c.replace(' ','')]-0.5)
                    y += eeg_part[c].mean()*(self.channel_pos.loc['y',c.replace(' ','')]-0.5)
                x /= len(self.channels)
                y /= len(self.channels)
                # 極座標変換
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                trial_features.extend([r, theta])

            # 特徴量4:ICA成分ごとのエネルギー
            info = mne.create_info(ch_names=self.channels.to_list(), sfreq=250, ch_types='eeg')
            raw = mne.io.RawArray(eeg.T, info, verbose=False)
            icaed = self.ica.get_sources(raw) 
            icaed = icaed.get_data().T  # shape = (250, n_components)
            for i in range(icaed.shape[1]):
                for icaed_part in np.array_split(icaed[:,i], reg_split):
                    norm = np.linalg.norm(icaed_part)
                    trial_features.append(norm)
            
            # 特徴量5:Discrete Wavelet Transformによる帯域ごとのエントロピー
            self.depth = 6 
            entropy_features = self.calculate_dwt_entropy(eeg.T.to_numpy(), level=self.depth, wavelet='db2')
            trial_features.extend(entropy_features.flatten())

            # 特徴量6:バンドパスフィルタによる帯域ごとのエネルギー
            band_energy = self.calculate_band_energy(eeg.T.to_numpy())
            trial_features.extend(band_energy.flatten())

            # 特徴量7:CSPコンポーネントの分散比
            for csp in self.CSPs:
                csped = csp.transform(np.expand_dims(eeg.T, axis=0))
                csped = csped[0] # shape = (n_components, 250)
                var_ratio_0 = np.var(csped[0])/(np.var(csped[0])+np.var(csped[-1]))
                var_ratio_l = np.var(csped[-1])/(np.var(csped[0])+np.var(csped[-1]))
                trial_features.extend([var_ratio_0, var_ratio_l])
 

            features.append(trial_features)


        # add feature names
        if add_feature_names:
            for lidx in range(len(self.labels)):
                for i in range(3):
                    self.feature_names.extend([f"dev_from{lidx}_{c}_{i}" for c in self.channels])
            for lidx in range(len(self.labels)):
                for i in range(reg_split):
                    self.feature_names.extend([f"corr_from{lidx}_{c}_{i}" for c in self.channels])
            for lidx in range(len(self.labels)):
                for c in self.channels:
                    self.feature_names.extend([f"dtw_from{lidx}_{c}"])
            for i in range(reg_split):
                self.feature_names.extend([f"slope_{c}_{i}" for c in self.channels])
                self.feature_names.extend([f"bias_{c}_{i}" for c in self.channels])
            for i in range(icaed.shape[1]):
                for j in range(reg_split):
                    self.feature_names.extend([f"ica_energy_{i}/{icaed.shape[1]}_{j}"])
            for c in self.channels:
                for j in range(4):
                    self.feature_names.extend([f"band_energy_{c}_{list(self.bands.keys())[j]}"])
            for l in self.labels:
                self.feature_names.extend([f"CSP{l}_var_first", f"CSP{l}_var_last"])

        if len(features[0]) != len(self.feature_names):
            print(f"feature length : {len(features[0])}, feature names length : {len(self.feature_names)}")
            raise ValueError("feature length and feature names length are not matched.")

        return features

    def moving_average(self, x, window):
        # windowは9とかでいい感じだった
        return np.convolve(x, np.ones(window), 'valid') / window

    def calculate_wpd_entropy(self,eeg_data, wavelet='haar', level=4, bins=10):
        """
        Calculate entropy of energy distributions for each node in WPD for each EEG channel.
        
        Parameters:
        - eeg_data: numpy array of shape (n_channels, n_samples), 72 channels in this case
        - wavelet: type of wavelet to use for WPD (default is 'haar')
        - level: level of Wavelet Packet Decomposition (default is 4)
        - bins: number of bins for histogram to estimate probability distribution (default is 40)
        
        Returns:
        - features: numpy array of shape (n_channels, 16), entropy values for each node
        """
        n_channels = eeg_data.shape[0]
        features = np.zeros((n_channels, 2 ** level))  # 16 nodes per channel for level 4

        for ch in range(n_channels):
            # Perform Wavelet Packet Decomposition
            wp = pywt.WaveletPacket(data=eeg_data[ch, :], wavelet=wavelet, mode='symmetric', maxlevel=level)
            
            # Extract energy time series and calculate entropy for each node
            for i, node in enumerate([node.path for node in wp.get_level(level, 'freq')]):
                coeffs = wp[node].data  # WPD coefficients at the node
                energy = np.square(coeffs)  # Energy is the square of the coefficients
                
                # Create a histogram to estimate probability distribution from energy time series
                hist, bin_edges = np.histogram(energy, bins=bins, density=True)
                
                # Avoid zero probabilities for entropy calculation
                hist = hist[hist > 0]
                
                # Calculate entropy using the estimated probability distribution
                features[ch, i] = entropy(hist)
        
        return features

    def calculate_dwt_entropy(self,eeg_data, wavelet='haar', level=6):
        """
        Calculate entropy for each channel and each DWT band in multi-channel EEG data.
        
        Parameters:
        eeg_data (numpy ndarray): 2D array of EEG data (channels x signal length).
        wavelet (str): The type of wavelet to use in DWT. Default is 'db4'.
        level (int): The level of DWT decomposition. Default is 4.
        
        Returns:
        band_entropies (numpy ndarray): 2D array of entropies (channels x number of bands).
        """
        num_channels, signal_length = eeg_data.shape
        band_entropies = np.zeros((num_channels, level + 1))  # +1 to include the approximation band
        
        for ch in range(num_channels):
            signal = eeg_data[ch]
            
            # Perform Discrete Wavelet Transform (DWT) for the current channel
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            
            # Calculate entropy for each band (approximation and details)
            for i, c in enumerate(coeffs):
                # Compute energy (square of the coefficients)
                energy = np.square(c)
                
                # Normalize energy to get a probability distribution
                prob_dist, _ = np.histogram(energy, bins=10, density=True)
                
                # Compute entropy using scipy's entropy function
                entropy_value = entropy(prob_dist + 1e-12)  # Small value added to avoid log(0)
                band_entropies[ch, i] = entropy_value
            
        
        return band_entropies

    def calculate_band_energy(self, eeg_data):
        '''
        Calculate energy for each channel and each band in multi-channel EEG data.

        Parameters:
        eeg_data (numpy ndarray): 2D array of EEG data (channels x signal length).

        Returns:
        band_energies (numpy ndarray): 2D array of energies (channels x number of bands).
        '''

        num_channels, signal_length = eeg_data.shape
        band_energies = np.zeros((num_channels, len(self.bands)))

        for ch in range(num_channels):
            signal = eeg_data[ch]
            for i, (band_name, band_range) in enumerate(self.bands.items()):
                filtered = self.bandpass_filter(signal, band_range[0], band_range[1])
                energy = np.sum(filtered**2)
                band_energies[ch, i] = energy

        return band_energies
    
    def bandpass_filter(self, x, lowcut, highcut, fs=500):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.lfilter(b, a, x)
