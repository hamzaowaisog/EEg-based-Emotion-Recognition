% Define input directories
original_dir = 'C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\cleaned_data';
auto_denoised_dir = 'C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\auto_denoised';
manual_denoised_dir = 'C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\manual_denoised'; % If available

% Get the list of .mat files in the original directory
files = dir(fullfile(original_dir, '*.mat'));

% Loop through each file and plot
for i = 1:length(files)
    % Load the original EEG data
    original_data = load(fullfile(original_dir, files(i).name));
    eeg_data = original_data.EEG;  % Assuming the variable is named 'EEG'
    
    % Load the auto-denoised EEG data
    auto_data = load(fullfile(auto_denoised_dir, files(i).name));
    eeg_auto_denoised = auto_data.eeg_denoised;  % Assuming the variable is named 'eeg_denoised'
    
    % Check if manually denoised data exists and load it
    manual_file = fullfile(manual_denoised_dir, files(i).name);
    if exist(manual_file, 'file')
        manual_data = load(manual_file);
        eeg_manual_denoised = manual_data.eeg_denoised;  % Assuming the variable is named 'eeg_denoised'
        manual_exists = true;
    else
        manual_exists = false;
    end
    
    % Plot the first 1000 time points of the data
    figure;
    
    % Plot original EEG data
    subplot(3, 1, 1);
    plot(eeg_data(:, 1:1000)');
    title('Original EEG Data');
    xlabel('Time Points');
    ylabel('Amplitude (\muV)');
    grid on;
    
    % Plot auto-denoised EEG data
    subplot(3, 1, 2);
    plot(eeg_auto_denoised(:, 1:1000)');
    title('Automatically Denoised EEG Data');
    xlabel('Time Points');
    ylabel('Amplitude (\muV)');
    grid on;
    
    % Plot manually denoised EEG data if available
    if manual_exists
        subplot(3, 1, 3);
        plot(eeg_manual_denoised(:, 1:1000)');
        title('Manually Denoised EEG Data');
        xlabel('Time Points');
        ylabel('Amplitude (\muV)');
        grid on;
    else
        subplot(3, 1, 3);
        text(0.5, 0.5, 'Manual Denoised Data Not Available', 'HorizontalAlignment', 'center', 'FontSize', 12);
        title('Manually Denoised EEG Data');
    end
    
    % Add overall title
    sgtitle(sprintf('Comparison of EEG Data for File: %s', files(i).name), 'Interpreter', 'none');
    
    % Pause to allow visualization before moving to the next file
    pause;
end
