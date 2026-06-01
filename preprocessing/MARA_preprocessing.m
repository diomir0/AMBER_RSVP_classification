% AMBER Dataset MARA Denoising
% Based on open EEG dataset: https://github.com/meharahsanawais/AMBER-EEG-Dataset.

% Awais MA, Redmond P, Ward TE and Healy G (2023). 
% AMBER: advancing multimodal brain-computer interfaces for enhanced robustness—A dataset for naturalistic settings.
% Front. Neuroergon. 4:1216440. 
% doi: 10.3389/fnrgo.2023.1216440
% https://www.frontiersin.org/journals/neuroergonomics/articles/10.3389/fnrgo.2023.1216440/

% Preprocessing (denoising + epoching)

%% Read the data
cd 'C:\Users\piton\Documents\Datasets\AMBER-EEG-Dataset\preprocessing'

% rawDir          = uigetdir([],'Path to the parent folder of raw EEG data');
rawDir = "C:\Users\piton\Documents\Datasets\AMBER-EEG-Dataset\Dataset\A_Raw";
raw_directory   = dir(rawDir);
subjectDirs     = raw_directory([raw_directory.isdir] & ...
    ~ismember({raw_directory.name}, [".", ".."]));
subjectDirs     = {subjectDirs.name}; 

SavePath        = [pwd '\clean\MARA'];
mkdir(SavePath)

ALLEEG = cell(1, numel(subjectDirs));

parfor p = 1:numel(subjectDirs)
    % EEG files
    subjectEEGData      = dir(fullfile(rawDir, subjectDirs{p}, '**\*.edf'));

    % Skip files without RSVP or P300 trials/markers
    patterns = ["B0", "B1", "B2", "X3", "X5", "X7"];
    keepIdx = ~contains({subjectEEGData.name}, patterns);
    subjectEEGData = subjectEEGData(keepIdx);
    subjectEEGFileNames = {subjectEEGData.name};

    % Marker files
    subjectMarkerData   = dir(fullfile(rawDir, subjectDirs{p}, '**\*.csv'));
    subjectMarkerFileNames = {subjectMarkerData.name};
    
    % --- Pre-allocate as Cell Arrays and Numeric Arrays ---
    numFiles = numel(subjectEEGFileNames);
    SubEEG = cell(1, numFiles); 
    initialVar_MARA = zeros(1, numFiles);
    denoisedVar_MARA = zeros(1, numFiles);
    time_MARA = zeros(1, numFiles);

    currentSubDir = subjectDirs{p};

    for f = 1:numFiles
        %% Initial Preprocessing
        EEGfilepath = fullfile(subjectEEGData(f).folder, subjectEEGFileNames{f});
        fName = subjectEEGFileNames{f};

        % --- Use a local 'EEG' variable, NOT a sliced structure ---
        EEG             = pop_biosig(EEGfilepath, 'channels', 1:1:32);
        EEG.setname     = fName(1:end-4);
        EEG.filename    = fName;
        EEG.subject     = subjectDirs{p};
        EEG.condition   = fName(9:10);
        EEG.session     = fName(7);

        EEG = pop_chanedit(EEG, {'lookup','standard_1005.elc'}, 'load', ...
            {'channel_locs_32set.loc','filetype','autodetect'});

        % Resample to 256 Hz
        EEG = pop_resample(EEG, 256);
    
        % Bandpass filter (1-40 Hz)
        EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 40);
    
        % Store initial variance
        initialVar_MARA(f) = mean(var(EEG.data, [], 1));

        % Bad channel rejection step prior to ICA (using ASR)
        original_channel_locations = EEG.chanlocs;
        EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion', 5, ...
            'ChannelCriterion', 0.8, 'LineNoiseCriterion', 4, ...
            'Highpass', 'off', 'BurstCriterion','off', ...
            'WindowCriterion', 'off', 'BurstRejection', 'off', ...
            'Distance', 'Euclidian');

        %% MARA Denoising
        % Run Infomax ICA Extended
        tic
        EEG = pop_runica(EEG, 'verbose', 'off', 'extended', 1, ...
            'interupt','off'); 
        
        % Use MARA to flag artifactual IComponents automatically
        [artifact_ICs, info] = MARA(EEG); 
        
        % Reject the ICs that MARA flagged as artifact
        EEG = pop_subcomp(EEG , artifact_ICs, 0); % Remove flagged components
        time_MARA(f) = toc;
        
        %% Bad Channel Interpolation
        EEG = pop_interp(EEG,  original_channel_locations, 'spherical'); 
        EEG = eeg_checkset(EEG);
        
        % Average variance across columns
        denoisedVar_MARA(f) = mean(var(EEG.data));  
        
        EEG = eeg_checkset(EEG);
        
        % Full rank average reference (following Makoto Miyakoshi's EEGlab plugin)
        EEG = GEDAI_nonRankDeficientAveRef(EEG);

        %% Epoching
        idx_file = find(strncmp(subjectMarkerFileNames, subjectEEGFileNames{f}, 10));
        if idx_file
            % Find the folder for the marker file
            idx_folder = find(strncmp({subjectMarkerData.folder}, subjectEEGData(f).folder, 80));
            markerFileName  = subjectMarkerFileNames{idx_file};
            markerFolder    = subjectMarkerData(idx_folder).folder; 
            markerFilepath  = fullfile(markerFolder, markerFileName);
            
            % Proceed to the epoching of the data
            EEG = amber_epoching(EEG, markerFilepath);
        else
            fprintf("WARNING: No markers file was found for this dataset.")
        end

        %% Saving
        new_filename = [subjectEEGFileNames{f}(1:end-4) '_MARA'];
        pop_saveset(EEG, 'filename', new_filename, 'filepath', SavePath);
        
        % Store the final result in the cell array
        SubEEG{f} = EEG;
    end
    SubEEG = [SubEEG{:}];  
    ALLEEG{p} = SubEEG;
end

ALLEEG = [ALLEEG{:}];
save("MARA_preproc.mat", "ALLEEG", "-v7.3");
