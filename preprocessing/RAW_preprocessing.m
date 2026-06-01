% AMBER Dataset RAW Preprocessing
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

SavePath        = [pwd '\clean\Raw'];
mkdir(SavePath)

ALLEEG = cell(1, numel(subjectDirs));

for p = 1:numel(subjectDirs)
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
    initialVar_Raw = zeros(1, numFiles);
    time_Raw = zeros(1, numFiles);

    currentSubDir = subjectDirs{p};

    parfor f = 1:numFiles
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
        initialVar_Raw(f) = mean(var(EEG.data, [], 1));

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
        new_filename = [subjectEEGFileNames{f}(1:end-4) '_Raw'];
        pop_saveset(EEG, 'filename', new_filename, 'filepath', SavePath);
        
        % Store the final result in the cell array
        SubEEG{f} = EEG;
    end
    SubEEG = [SubEEG{:}];  
    ALLEEG{p} = SubEEG;
end

ALLEEG = [ALLEEG{:}];
save("Raw_preproc.mat", "ALLEEG", "-v7.3");
