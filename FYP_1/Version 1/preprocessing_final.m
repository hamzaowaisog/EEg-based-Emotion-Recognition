% Preprocessing Script for SEED Dataset in MATLAB with Channel Order and Time Segmentation

% Load the raw SEED dataset
% Replace "your_file_path" with the actual file path
addpath('E:\FYP\fieldtrip-master');
addpath('E:\FYP\NoiseTools');

ft_defaults;
nt_greetings;

cfg = [];
cfg.dataset = 'E:\FYP\Egg-Based Emotion Recognition\EEg-based-Emotion-Recognition\SEED\SEED_EEG\SEED_RAW_EEG\1_1.cnt';
EEG_data = ft_preprocessing(cfg);

% Load channel order from channel-order.xlsx
% Read the channel labels from the first sheet, first column
channel_order_table = readtable('channel-order.xlsx', 'Sheet', 'Sheet1', 'Range', 'A:A');
channel_order = channel_order_table{:, 1}; % Extract the channel labels as a cell array

% Reorder channels according to the provided channel order using FieldTrip
cfg = [];
cfg.channel = channel_order;
EEG_data = ft_selectdata(cfg, EEG_data);

% Step 1: Bandpass Filtering (0.05 - 47 Hz) using FieldTrip
cfg = [];
cfg.bpfilter = 'yes';
cfg.bpfreq = [0.05 47];
EEG_filtered = ft_preprocessing(cfg, EEG_data);

% Step 2: Independent Component Analysis (ICA) for Artifact Removal using FieldTrip
cfg = [];
cfg.method = 'runica'; % Runica is a popular method for ICA
EEG_ica = ft_componentanalysis(cfg, EEG_filtered);

% Identify components related to eye movements or muscle artifacts
% This step usually requires visual inspection and manual rejection
cfg = [];
cfg.component = [1:20]; % Specify the components you want to inspect
ft_topoplotIC(cfg, EEG_ica); % Plot components for inspection

% Remove identified artifacts
cfg = [];
cfg.component = [1, 2, 3]; % Replace with components identified as artifacts
EEG_clean = ft_rejectcomponent(cfg, EEG_ica, EEG_filtered);

% Step 3: Bandpass Filtering Again (4 - 47 Hz) using FieldTrip
cfg = [];
cfg.bpfilter = 'yes';
cfg.bpfreq = [4 47];
EEG_filtered_final = ft_preprocessing(cfg, EEG_clean);

% Step 4: Re-reference to Common Average
cfg = [];
cfg.reref = 'yes';
cfg.refchannel = 'all';
EEG_reref = ft_preprocessing(cfg, EEG_filtered_final);

% Step 5: Load Time Points and Segment Data
% Load time points from time.txt
start_point_list = [27000,290000,551000,784000,1050000,1262000,1484000,1748000,1993000,2287000,2551000,2812000,3072000,3335000,3599000]; % in samples, from time.txt
end_point_list = [262000,523000,757000,1022000,1235000,1457000,1721000,1964000,2258000,2524000,2786000,3045000,3307000,3573000,3805000];

% Segment the data into epochs using start and end points
n_epochs = length(start_point_list);
epochs = cell(1, n_epochs); % Store epochs in a cell array

for i = 1:n_epochs
    epochs{i} = EEG_reref.trial{1}(:, start_point_list(i):end_point_list(i));
end

% Step 6: Feature Extraction - Differential Entropy (DE)
DE_features = [];
frequency_bands = {'theta', [4, 8]; 'alpha', [8, 13]; 'beta', [13, 30]; 'gamma', [30, 47]};

for i = 1:size(frequency_bands, 1)
    band = frequency_bands{i, 2};
    [b, a] = butter(4, band / (EEG_reref.fsample / 2), 'bandpass');
    for j = 1:n_epochs
        data = filtfilt(b, a, epochs{j}'); % Transpose data for filtering
        data = data'; % Transpose back after filtering
        DE = 0.5 * log(2 * pi * exp(1) * var(data, 0, 2)); % Differential Entropy calculation
        DE_features = [DE_features, DE];
    end
end

% Step 7: Normalization and Smoothing
% Adaptive z-score normalization and smoothing with Linear Dynamical System
mean_train = mean(DE_features, 2);
std_train = std(DE_features, 0, 2);
normalized_DE = (DE_features - mean_train) ./ std_train;
smoothed_DE = smoothdata(normalized_DE, 'movmean', 5); % Moving average with window of 5

% Save the preprocessed data
save('preprocessed_SEED_data.mat', 'smoothed_DE');
