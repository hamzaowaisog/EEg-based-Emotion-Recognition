% Define the directory containing denoised EEG data
denoised_dir = 'C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\auto_denoised';

% Get the list of .mat files in the denoised directory
files = dir(fullfile(denoised_dir, '*.mat'));

% Check if there are at least 4 files
num_files = length(files);
if num_files < 4
    error('Not enough files to select 4 random samples. Found only %d files.', num_files);
end

% Randomly select 4 unique indices
rand_indices = randperm(num_files, 4);

% Loop through the selected files and visualize them
for i = 1:4
    % Load the denoised EEG data
    data = load(fullfile(denoised_dir, files(rand_indices(i)).name));
    eeg_denoised = data.eeg_denoised;  % Assuming the variable is named 'eeg_denoised'

    % Check the size of the data
    disp(size(eeg_denoised));

    % Plot each channel with an offset for better visualization
    figure;
    num_channels = size(eeg_denoised, 1);
    time_points = size(eeg_denoised, 2);
    offset = 1000;  % Adjust the offset value as needed for clear separation

    hold on;
    for ch = 1:num_channels
        plot(1:time_points, eeg_denoised(ch, :) + (ch - 1) * offset, 'DisplayName', sprintf('Ch%d', ch));
    end
    hold off;

    % Add title, labels, and legend
    title(sprintf('Denoised EEG Data - File %s', files(rand_indices(i)).name), 'Interpreter', 'none');
    xlabel('Samples');
    ylabel('Amplitude (µV) with Offsets');
    legend('show');
    grid on;
end
