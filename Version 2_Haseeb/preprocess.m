% Preprocessing Script for All SEED Dataset Files in a Folder
% Ensure Fieldtrip and NoiseTools are added to your MATLAB path

addpath('D:\FAST\FYP\fieldtrip-master');
ft_defaults;  % Initialize Fieldtrip
addpath('D:\FAST\FYP\NoiseTools');

% SEED Dataset Preprocessing Script
% Ensure Fieldtrip and NoiseTools are added to your MATLAB path

% Define the path to the Preprocessed_EEG folder and output folder
data_folder = 'D:\FAST\EEg-based-Emotion-Recognition\Preprocessed_EEG';
output_folder = 'D:\FAST\EEg-based-Emotion-Recognition\Preprocessed_Output';

% Create the output folder if it doesn't exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% List all .mat files in the folder
mat_files = dir(fullfile(data_folder, '*.mat'));

% Sampling rate (assuming 200 Hz for SEED dataset)
fs = 200;

% Loop through each .mat file
for i = 1:length(mat_files)
    file_path = fullfile(data_folder, mat_files(i).name);
    fprintf('\nProcessing file: %s\n', mat_files(i).name);
    
    % Load the .mat file
    data = load(file_path);
    var_names = fieldnames(data);
    
    % Initialize a struct to store preprocessed data
    preprocessed_data = struct();
    
    % Loop through each EEG variable (ww_eeg1, ww_eeg2, ..., ww_eeg15)
    for j = 1:length(var_names)
        eeg_data = data.(var_names{j});  % Load the EEG matrix (62 × N)
        
        % Transpose data to (time points × channels) for processing
        eeg_data = eeg_data';  % Now it's N × 62
        
        %% Step 1: Bandpass Filter (4 to 47 Hz)
        fprintf('Applying bandpass filter to %s...\n', var_names{j});
        eeg_filtered = bandpass(eeg_data, [4 47], fs);
        
        %% Step 2: Denoising with NoiseTools
        fprintf('Applying denoising to %s...\n', var_names{j});
        window_size = 5;  % Window size for median filtering
        eeg_denoised = medfilt1(eeg_filtered, window_size, [], 2);
        
        %% Step 3: Re-referencing to Common Average
        fprintf('Re-referencing to common average for %s...\n', var_names{j});
        eeg_ref = eeg_denoised - mean(eeg_denoised, 2);  % Subtract the mean across channels
        
        %% Store the preprocessed data
        preprocessed_data.(var_names{j}) = eeg_ref';
    end
    
    %% Save the preprocessed data to a new .mat file
    output_file = fullfile(output_folder, ['preprocessed_', mat_files(i).name]);
    save(output_file, '-struct', 'preprocessed_data');
    fprintf('Preprocessed data saved to: %s\n', output_file);
end

disp('All files have been processed and saved.');
