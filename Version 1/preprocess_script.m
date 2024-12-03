%Add Fieldtrip and NoiseTools toolkits to the MATLAB path
addpath('D:\FAST\FYP\fieldtrip-master');
addpath('D:\FAST\FYP\NoiseTools');
ft_defaults; % Initialize Fieldtrip

% Directory containing the .cnt files
inputDir = 'data/';

% Directory to save preprocessed .mat files
outputDir = 'processed/';

% Get list of .cnt files
cntFiles = dir(fullfile(inputDir, '*.cnt'));
fprintf('Found %d .cnt files\n', length(cntFiles));

for i = 1:length(cntFiles)
    % Load the .cnt file
    cfg = [];
    cfg.dataset = fullfile(inputDir, cntFiles(i).name);
    data = ft_preprocessing(cfg);

    % Step 1: Downsample data to 200 Hz (if not already done by the provider)
    cfg = [];
    cfg.resamplefs = 200;
    cfg.detrend = 'no';
    data = ft_resampledata(cfg, data);

    %Step 2: Apply Initial Bandpass Filter (0.05 to 75 Hz)
    cfg = [];
    cfg.bpfilter = 'yes';
    cfg.bpfreq = [0.05 75];
    data = ft_preprocessing(cfg, data);

    % Step 3: Apply ICA for artifact removal
    cfg = [];
    cfg.method = 'runica'; %uses the Infomax algorithm
    %Approach 1 (currently)
    %cfg.runica.pca = min(62, length(data.label)-1); %set the number of principal components to retain
    %Approach 2
    %cfg.runica.pca = min(size(data.trial{1}, 1), 64); % Limit components
    comp = ft_componentanalysis(cfg, data);

    %Viusalize the components and select the ones to remove
    %Approach 1 (not using)
    %ft_databrowser([], comp);
    %cfg=[];
    %cfg.component = [1, 2]; % Replace with identified artifact components
    %data = ft_rejectcomponent(cfg, comp, data);
    %Approach 2
    %cfg = [];
    %cfg.component = 1:20;  % Adjust as needed
    %ft_topoplotIC(cfg, comp);
    % Uncomment below to see time course of components
    %cfg = [];
    %cfg.layout = 'biosemi64.lay';
    %ft_databrowser(cfg, comp);

    % Manually remove artifact components (use output from visual inspection)
    %artifact_components = [1, 2];  % Replace with component indices to remove
    %cfg = [];
    %cfg.component = artifact_components;
    %data = ft_rejectcomponent(cfg, comp, data);
    %Approach 3
    %cfg = [];
    %cfg.layout = 'biosemi64.lay'; %specify layout if available
    %ft_databrowser(cfg, comp);


    % Step 4: Automatic Denoising with NoiseTools
    %Approach 1
    %eeg_data = double(cat(2,data.trial{:}));
    %nt_data = nt_denoise(eeg_data);

    % Fix noisy channels using interpolation
    noisy_threshold = 0.3; % Define 30% threshold for noisy channels
    num_trials = length(data.trial);
    interpolated_trials = 0;

    for trial_idx = 1:num_trials
        trial_data = data.trial{trial_idx};
        num_channels = size(trial_data, 1);

        for chan_idx = 1:num_channels
            channel_data = trial_data(chan_idx, :);
            outliers = abs(channel_data) > 3 * median(abs(channel_data));
            if sum(outliers) / length(channel_data) > noisy_threshold
                % Interpolate using 3 closest channels
                if chan_idx > 1 && chan_idx < num_channels
                    trial_data(chan_idx, :) = mean([trial_data(chan_idx-1, :); trial_data(chan_idx+1, :)], 1);
                elseif chan_idx == 1
                    trial_data(chan_idx, :) = trial_data(chan_idx+1, :);
                else
                    trial_data(chan_idx, :) = trial_data(chan_idx-1, :);
                end
                interpolated_trials = interpolated_trials + 1;
            end

            % Additional outlier correction
            diff_data = abs(diff(channel_data));
            large_diff = find(diff_data > 100);
            for idx = 1:length(large_diff)
                trial_data(chan_idx, large_diff(idx) + 1) = trial_data(chan_idx, large_diff(idx));
            end
        end
        data.trial{trial_idx} = trial_data;
    end

    fprintf('Number of interpolated trials: %d\n', interpolated_trials);%

    % Step 6: Apply final bandpass filter (4 to 47 Hz)
    cfg = [];
    cfg.bpfilter = 'yes';
    cfg.bpfreq = [4 47]; % Filter range: 4 to 47 Hz
    cfg.demean = 'yes';
    data = ft_preprocessing(cfg, data);

    % Step 7: Re-reference data to common average
    cfg = [];
    cfg.reref = 'yes';
    cfg.refchannel = 'all';
    data = ft_preprocessing(cfg, data);

    % Step 8: Save preprocessed data
    [~, name, ~] = fileparts(cntFiles(i).name);
    outputFilename = fullfile(outputDir, [name, '_preprocessed.mat']);
    save(outputFilename, 'data');

    fprintf('Processed and saved: %s\n', outputFilename);

end
%Approach 2

%nt_data = nt_mmat2nt(data.trial);
% Interpolate noisy channels (threshold: >30% outliers)
% Outliers are defined as values exceeding 3x median absolute value
%nt_data = nt_interp_bad_channels(nt_data, 0.3);

% Fix remaining outliers by threshold
%for ch = 1:size(nt_data, 2)
%    diff_vals = abs(diff(nt_data(:, ch)));
%    outliers = find(diff_vals > outlier_threshold);
%    nt_data(outliers + 1, ch) = nt_data(outliers, ch);  % Replace with previous value
%end

% Convert NoiseTools output back to Fieldtrip format
%data.trial{1} = nt_data;

% Step 4: Re-reference data to the common average
%cfg = [];
%cfg.reref = 'yes';
%cfg.refchannel = 'all';
%data = ft_preprocessing(cfg, data);

% Save the preprocessed data
%save(fullfile(output_dir, [file_name(1:end-4), '.mat']), 'data');
