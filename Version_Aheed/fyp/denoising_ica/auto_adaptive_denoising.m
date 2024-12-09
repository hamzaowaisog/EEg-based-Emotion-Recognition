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

    % Multiple DSS iterations
    for iter = 1:3
        % Calculate the biased covariance of the EEG data
        c1 = cov(eeg_data');  % Biased covariance

        % Define the baseline covariance (identity matrix)
        c0 = eye(size(c1));

        % Compute DSS using nt_dss0
        [todss, fromdss, ratio] = nt_dss0(c0, c1);

        % Project data to DSS space
        dss_components = (eeg_data' * todss)';

        % Adaptive threshold for artifact detection
        % Adaptive threshold for artifact detection
        threshold = mean(max(abs(dss_components), [], 2)) + 2 * std(max(abs(dss_components), [], 2));

        artifact_indices = find(max(abs(dss_components), [], 2) > threshold);
        fprintf('Iteration %d - Removing artifact components: %s\n', iter, mat2str(artifact_indices));

        % Zero out the detected artifact components
        if ~isempty(artifact_indices)
            dss_components(artifact_indices, :) = 0;
        end

        % Reconstruct cleaned EEG data
        eeg_data = fromdss * dss_components;
    end

    % Save the denoised EEG data
    eeg_denoised = eeg_data;
    output_file = fullfile(output_dir, files(i).name);
    save(output_file, 'eeg_denoised');
    fprintf('Processed and saved %s\n\n', files(i).name);
end
