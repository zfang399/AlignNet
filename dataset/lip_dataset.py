# Finished by zfang 2020/02/15 15:19pm
import torch.utils.data as data
import torch
import numpy as np
import random
import scipy
import scipy.io
from scipy import interpolate
from scipy import signal
import torchaudio
from sklearn.preprocessing import MinMaxScaler
import librosa

def read_sample_list(sample_txt_path):
    # get sample list from file
    sample_list = []
    input_file = open(sample_txt_path)
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1] == '\n':
            input_line = input_line[:-1]
        # strip .mat format
        sample_list.append(input_line[:-4])
    input_file.close()
    
    return sample_list

class VoxCelebDataset(data.Dataset):
    def __init__(self, args, cfg, mode = "train"):
        # setup from configuration
        self.cfg = cfg
        self.args = args
        self.mode = mode
        if self.mode == 'train':
            self.sample_txt_path = self.cfg["train_txt_path"]
        elif self.mode == 'val':
            self.sample_txt_path = self.cfg["val_txt_path"]
        elif self.mode == 'test':
            self.sample_txt_path = self.cfg["test_txt_path"]

        self.sample_list = read_sample_list(self.sample_txt_path)
        random.shuffle(self.sample_list)
        self.length = len(self.sample_list)
        
        self.batch_size = self.cfg["batch_size"]
        
        # distortion / stretch / global_shift settings
        self.local_distortion = self.args.local_distortion
        if self.local_distortion:
            self.max_distortion_ratio = 0.5 * self.cfg["max_distortion"] / self.cfg["sample_duration"]
        self.global_stretch = self.args.global_stretch
        if self.global_stretch:
            self.min_stretch_frames = int(round(self.cfg["min_stretch"]*self.cfg["vid_framerate"]))
            self.max_stretch_frames = int(round(self.cfg["max_stretch"]*self.cfg["vid_framerate"]))
        self.global_shift = self.args.global_shift
        if self.global_shift:
            self.max_shift_frames = int(round(self.cfg["max_shift"]*self.cfg["vid_framerate"]))
        
        # sample_count for changing sample length (if global_stretch is True) each batch
        self.sample_count = 0
        self.frames = int(self.cfg["sample_duration"]*self.cfg["vid_framerate"])

        # setup spectrogram functions
        self.calc_spec = torchaudio.transforms.MelSpectrogram(sample_rate = self.cfg["mel_sr"], 
                                                              n_fft = self.cfg["mel_n_fft"], 
                                                              hop_length = self.cfg["mel_hop"], 
                                                              f_min = self.cfg["f_min"],
                                                              f_max = self.cfg["f_max"],
                                                              n_mels = self.cfg["num_mels"])
        self.spec2db = torchaudio.transforms.AmplitudeToDB()

    def __getitem__(self, index):
        np.random.seed()
        sample_name = self.sample_list[index]
        
        ###############################################################
        # 0. Batch settings
        ###############################################################
        if self.sample_count % self.batch_size == 0:
            self.sample_count = 0
            # random global stretch & compression for the batch
            if self.global_stretch:
                self.frames = np.random.randint(self.min_stretch_frames, self.max_stretch_frames+1)
        self.sample_count += 1
        target = np.linspace(-1,1,int(self.cfg["sample_duration"]*self.cfg["vid_framerate"]))

        ###############################################################
        # 1. Video data (keypoints)
        ###############################################################
        video_feature = scipy.io.loadmat('{0}/{1}.mat'.format(self.cfg["video_feature_root"], sample_name))
        
        # grab keypoints (discard foot keypoints)
        video_feature = video_feature['lip_list'][0]
        video_feature = video_feature[:,0:self.cfg["num_lip_keypoints"],:]

        # crop desired length from the video --- 300 frames for 12 seconds
        n_frames = video_feature.shape[0]
        frame_start = np.random.randint(0, n_frames - self.cfg["sample_duration"] * self.cfg["vid_framerate"] + 1)
        frame_end = frame_start + self.cfg["sample_duration"] * self.cfg["vid_framerate"]
        video_feature = video_feature[frame_start:frame_end,:,:]
        
        # normalize keypoint positions
        for kp in range(self.cfg["num_lip_keypoints"]):
            scaler = MinMaxScaler(feature_range = (-1,1))
            video_feature[:, kp, :] = scaler.fit_transform(video_feature[:, kp, :])
        
        #video_feature = video_feature.reshape(-1, self.cfg["num_lip_keypoints"]*2)
        
        # video augmentation --- horizontal flipping
        if self.mode == 'train':
            if np.random.rand() < self.cfg["prob_horizontal_flip"]:
                video_feature[:,:,0] = video_feature[:,:,0] * -1

        ###############################################################
        # 2. Global Shift
        ###############################################################
        if self.global_shift:
            # shift the audio in feasible range
            left_shift = min(frame_start, self.max_shift_frames)
            right_shift = min(n_frames - frame_end, self.max_shift_frames)
            frame_shift = np.random.randint(-left_shift, right_shift+1)
            audio_frame_start = frame_start + frame_shift
            # modify the target array
            target += 2 * frame_shift/(self.cfg["vid_framerate"]*self.cfg["sample_duration"])
        else:
            audio_frame_start = frame_start
            
        ###############################################################
        # 3. Audio data (Spectrogram)
        ###############################################################
        waveform, audio_sample_rate = torchaudio.load('{0}/{1}.mp3'.format(self.cfg["audio_feature_root"], sample_name))
        # crop 12s of audio data
        audio_start = int(audio_frame_start * self.cfg["mel_sr"] / self.cfg["vid_framerate"])
        audio_end = int(audio_start + self.cfg["mel_sr"] * self.cfg["sample_duration"])
        waveform = waveform[0, audio_start:audio_end]

        # compute spectrogram
        audio_feature = self.spec2db(self.calc_spec(waveform))
        audio_feature = torch.squeeze(audio_feature, 0)

        # normalize spectrogram
        audio_feature = (audio_feature - torch.mean(audio_feature)) / (torch.max(audio_feature) - torch.min(audio_feature))
        
        # spectrogram augmentation
        if self.mode == 'train':
            # frequency masking
            if np.random.rand() < self.cfg["prob_freqmask"]:
                dur = np.random.randint(self.cfg["min_freqmask"], self.cfg["max_freqmask"]+1)
                st = np.random.randint(0,audio_feature.shape[0] - dur + 1)
                audio_feature[st:st+dur,:] = 0
            
            # time masking
            if np.random.rand() < self.cfg["prob_timemask"]:
                dur = np.random.randint(self.cfg["min_timemask"], self.cfg["max_timemask"]+1)
                st = np.random.randint(0,audio_feature.shape[1] - dur + 1)
                audio_feature[:,st:st+dur] = 0
        
        ###############################################################
        # 3. Global Stretch & Local Distortion
        ###############################################################
        new_video_feature = np.zeros((self.frames, self.cfg["num_lip_keypoints"],2))
        new_target = np.zeros(self.frames)
        
        # Random distortion
        random_position = np.linspace(-1,1,self.frames)
        if self.local_distortion:
            random_position = np.zeros(self.frames)
            while np.max(np.abs(random_position - np.linspace(-1,1,self.frames))) > self.max_distortion_ratio:
                resample_len = int(self.frames * self.cfg["random_resample_rate"])
                random_position = np.random.rand(resample_len)
                random_position = np.sort(
                    (random_position - np.min(random_position)) * 2 /
                    (np.max(random_position) - np.min(random_position)) - 1)
                f = interpolate.interp1d(np.linspace(-1, 1, resample_len), random_position, kind='linear')
                random_position = f(np.linspace(-1, 1, self.frames))
        
        # Distorted & Stretched video feature
        for k in range(self.frames):
            orig_index = (random_position[k] + 1) / 2 * (self.cfg["sample_duration"] * self.cfg["vid_framerate"] - 1)
            lower = int(np.floor(orig_index))
            upper = int(np.ceil(orig_index))
            
            if lower == upper:
                new_video_feature[k,:,:] = video_feature[lower,:,:]
                new_target[k] = target[lower]
            else:
                new_video_feature[k,:,:] = video_feature[lower,:,:]*(upper-orig_index) + video_feature[upper,:,:]*(orig_index-lower)
                new_target[k] = target[lower]*(upper-orig_index) + target[upper]*(orig_index-lower)
        video_feature = new_video_feature
        target = new_target
        
        # velocity
        video_velo = np.zeros((self.frames, self.cfg["num_lip_keypoints"],2))
        video_velo[1:,:,:] = video_feature[1:,:,:] - video_feature[:-1,:,:]
        video_velo = video_velo / np.amax(np.absolute(video_velo))
        
        # acceleration
        video_acc = np.zeros((self.frames, self.cfg["num_lip_keypoints"],2))
        video_acc[1:,:,:] = video_velo[1:,:,:] - video_velo[:-1,:,:]
        video_acc = video_acc / np.amax(np.absolute(video_acc))
        
        # aggregate
        video_agg = np.zeros((self.frames, self.cfg["num_lip_keypoints"],2,2))
        video_agg[:,:,:,0] = video_velo
        video_agg[:,:,:,1] = video_acc

        return {"video_feature": torch.from_numpy(video_agg.astype(np.float32)), \
                "audio_feature": audio_feature, \
                "target": torch.from_numpy(target.astype(np.float32)), \
                "sample_name": sample_name}

    def __len__(self):
        return self.length
