function EEG = amber_epoching(EEG, markerFilepath)
    % --- Ensure markers is a matrix ---
    markers = readmatrix(markerFilepath);
    if istable(markers)
        markers = table2array(markers); % Convert to matrix if it's a table
    end

    % Identify indices that are not 'Impedance'
    validIdx = find(~strcmp({EEG.event.type}, 'Impedance'));

    % Map markers to existing events only up to the minimum of the two lengths
    % This prevents the "Dimension Mismatch" crash
    numToMap = min(length(validIdx), length(markers));
    for i = 1:numToMap
        EEG.event(validIdx(i)).type = markers(i);
    end

    % Now remove the remaining 'Impedance' markers so the list is clean
    % EEG.event(~strcmp({EEG.event.type}, 'Impedance')) = [];
    % Note: The above line can be tricky. Better to filter:
    tmpEvents = EEG.event;
    keepIdx = ~strcmp({tmpEvents.type}, 'Impedance');
    EEG.event = tmpEvents(keepIdx);

    % --- BEGIN README SURGERY ---
    % General RSVP Rule: Discard first and last triggers
    % if length(EEG.event) >= 2
    %     EEG.event(1) = [];
    %     EEG.event(end) = [];
    % end

    % Participant 7 || Session 1 || X1
    if (contains(markerFilepath, 'P07') && contains(markerFilepath, 'Ss1-X1') && ...
            length(EEG.event)~=360)
        targets = [116, 180];
        for t = targets
            if t < length(EEG.event)
                newEv = EEG.event(t);
                newEv.latency = (EEG.event(t).latency + EEG.event(t+1).latency)/2;
                % Assign a type for the new event (from markers list)
                if length(markers) >= t+1, newEv.type = markers(t+1); end
                EEG.event = [EEG.event(1:t), newEv, EEG.event(t+1:end)];
            end
        end
    end

    % Participant 7 || Session 1 || X8
    if (contains(markerFilepath, 'P07') && contains(markerFilepath, 'Ss1-X8') &&...
            length(EEG.event)~=360)
        targets = [49, 50, 139];
        for t = targets
            if t < length(EEG.event)
                newEv = EEG.event(t);
                newEv.latency = (EEG.event(t).latency + EEG.event(t+1).latency)/2;
                if length(markers) >= t+1, newEv.type = markers(t+1); end
                EEG.event = [EEG.event(1:t), newEv, EEG.event(t+1:end)];
            end
        end
    end

    % Participant 5 || Session 4 || X1
    if (contains(markerFilepath, 'P05') && contains(markerFilepath, 'Ss4-X1') && ...
            length(EEG.event)~=360)
        if length(EEG.event) > 8
            EEG.event(end-7:end) = [];
        end
        if length(EEG.event) >= 184
            EEG.event(184) = [];
        end
    end

    % Participant 8 || Session 1 || X4
    if (contains(markerFilepath, 'P08') && contains(markerFilepath, 'Ss1-X4') && ...
            length(EEG.event)~=360)
        fprintf(string(length(EEG.event)));
        if length(EEG.event) >= 199
            EEG.event(199) = [];
        end
    end

    idx = ~strcmp({EEG.event.type}, 'Impedance');
    cellMarkers = num2cell(markers);
    [EEG.event(idx).type] = cellMarkers{:};

    % Epoch extraction (time window defined in paper)
    EEG = pop_epoch(EEG, {1 2}, [-0.2 0.8], 'epochinfo', 'yes');


    % Strip out foreign events from each epoch
    for i = 1:EEG.trials
        % Find all events belonging to the current epoch
        epoch_events = find([EEG.event.epoch] == i);
        
        % If there are overlapping foreign events (more than 1 event in this epoch)
        if length(epoch_events) > 1
            % Keep the first event; mark all subsequent events for deletion
            foreign_event_indices = epoch_events(2:end);
            
            % Set their epoch markers to 0 so they dissociate from this trial
            [EEG.event(foreign_event_indices).epoch] = deal(0);
        end
    end

    EEG.event([EEG.event.epoch] == 0) = [];
    EEG = eeg_checkset(EEG, 'eventconsistency');

end
