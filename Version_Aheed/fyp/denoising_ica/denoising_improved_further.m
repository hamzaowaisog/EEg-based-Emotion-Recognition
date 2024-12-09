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

    % Plot DSS components for visual inspection
    figure;
    plot(dss_components');
    title(sprintf('DSS Components - File %s', files(i).name), 'Interpreter', 'none');
    xlabel('Samples');
    ylabel('Amplitude');
    grid on;

    % Allow user to select artifact components manually
    prompt = 'Enter artifact component indices (e.g., [1, 2]): ';
    artifact_indices = input(prompt);

    % Zero out the detected artifact components
    if ~isempty(artifact_indices)
        dss_components(artifact_indices, :) = 0;
        fprintf('Removed artifact components: %s\n', mat2str(artifact_indices));
    else
        fprintf('No components removed for this file.\n');
    end

    % Reconstruct cleaned EEG data
    eeg_denoised = fromdss * dss_components;  % Reconstruct back to original shape

    % Ensure the reconstructed data has the original dimensions
    if size(eeg_denoised, 1) ~= num_channels
        eeg_denoised = eeg_denoised(1:num_channels, :);
    end

    fprintf('Denoised size: [%d %d]\n', size(eeg_denoised, 1), size(eeg_denoised, 2));

    % Save the denoised EEG data
    output_file = fullfile(output_dir, files(i).name);
    save(output_file, 'eeg_denoised');
    fprintf('Processed and saved %s\n\n', files(i).name);

    % Close the figure to avoid clutter
    close all;
end
