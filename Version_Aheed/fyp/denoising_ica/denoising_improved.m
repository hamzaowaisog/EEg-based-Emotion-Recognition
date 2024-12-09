% Define input and output directories
input_dir = 'C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\cleaned_data';
output_dir = 'C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\denoised_data';

% Create the output directory if it doesn't exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Get the list of .mat files in the input directory
files = dir(fullfile(input_dir, '*.mat'));

% Loop through each file
for i = 1:length(files)
    % Load the EEG data
    data = load(fullfile(input_dir, files(i).name));
    eeg_data = data.EEG;  % Assuming the variable is named 'EEG'

    % Calculate the biased covariance of the EEG data
    c1 = cov(eeg_data');  % Biased covariance (transpose for channels × samples)

    % Define the baseline covariance (identity matrix)
    c0 = eye(size(c1));   % Identity matrix as the baseline covariance

    % Compute DSS using nt_dss0
    [todss, fromdss, ratio] = nt_dss0(c0, c1);

    % Project data to DSS space
    dss_components = (eeg_data' * todss)';  % Project to DSS space

    % Plot DSS components for visual inspection
    figure;
    plot(dss_components');
    title(sprintf('DSS Components for File %d', i));
    xlabel('Samples');
    ylabel('Amplitude');
    pause(2);  % Pause to allow inspection; adjust as needed

    % Inspect and manually select artifact components to remove
    artifact_indices = [1, 2, 4];  % Adjust based on visual inspection of DSS plot

    % Zero out the artifact components
    dss_components(artifact_indices, :) = 0;

    % Reconstruct cleaned EEG data
    eeg_denoised = fromdss * dss_components;  % Corrected reconstruction step

    % Save the denoised EEG data
    output_file = fullfile(output_dir, files(i).name);
    save(output_file, 'eeg_denoised');
    fprintf('Processed and saved %s\n', files(i).name);
end
