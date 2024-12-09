% Define input and output directories
input_dir = 'C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\cleaned_data';
output_dir = 'C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\auto_denoised';

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
    num_channels = size(eeg_data, 1);
    fprintf('Processing File: %s\n', files(i).name);
    fprintf('Original size: [%d %d]\n', num_channels, size(eeg_data, 2));

    % Calculate the biased covariance of the EEG data
    c1 = cov(eeg_data');  % Biased covariance (transpose for channels × samples)

    % Define the baseline covariance (identity matrix)
    c0 = eye(size(c1));   % Identity matrix as the baseline covariance

    % Compute DSS using nt_dss0
    [todss, fromdss, ratio] = nt_dss0(c0, c1);

    % Ensure fromdss has the correct number of rows (expand if necessary)
    if size(fromdss, 1) ~= num_channels
        fromdss = [fromdss; zeros(num_channels - size(fromdss, 1), size(fromdss, 2))];
    end

    % Project data to DSS space
    dss_components = (eeg_data' * todss)';  % Project to DSS space

    % Automatically detect artifact components based on threshold
    threshold = 800;  % Adjust this threshold based on your data
    artifact_indices = find(max(abs(dss_components), [], 2) > threshold);
    fprintf('Removing artifact components: %s\n', mat2str(artifact_indices));


    % Zero out the detected artifact components
    if ~isempty(artifact_indices)
        dss_components(artifact_indices, :) = 0;
        fprintf('Automatically removed artifact components: %s\n', mat2str(artifact_indices));
    else
        fprintf('No artifacts detected for this file.\n');
    end

    % Reconstruct cleaned EEG data
    eeg_denoised = fromdss * dss_components;  % Reconstruct back to original shape

    % Ensure the reconstructed data has the original dimensions
    if size(eeg_denoised, 1) ~= num_channels
        eeg_denoised = eeg_denoised(1:num_channels, :);
    end

    fprintf('Denoised size: [%d %d]\n', size(eeg_denoised, 1), size(eeg_denoised, 2));
    fprintf('Size of todss: [%d %d]\n', size(todss));
    fprintf('Size of fromdss: [%d %d]\n', size(fromdss));


    % Save the denoised EEG data
    output_file = fullfile(output_dir, files(i).name);
    save(output_file, 'eeg_denoised');
    fprintf('Processed and saved %s\n\n', files(i).name);
end
