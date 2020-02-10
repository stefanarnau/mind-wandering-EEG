

% ======================= GENERAL INFO ====================================
%
% This is the Matlab code used for analyzing the data for the study  "Inter-Trial
% Alpha Power Indicates Mind Wandering"
%
% Code by Stefan Arnau, January 2020
% Email: arnau@ifado.de
% GitHub: github.com/fischmechanik/mind-wandering-EEG
% OSF repository: (WILL BE PROVIDED WHEN ACCEPTED/PUBLISHED)
%

% Remove evil residuals
clear all;

% Path vars (insert paths here)
PATH_EEGLAB           = PATH_PLACEHOLDER; 
PATH_FIELDTRIP        = PATH_PLACEHOLDER; 
PATH_RAW_DATA         = PATH_PLACEHOLDER; 
PATH_ICSET            = PATH_PLACEHOLDER; 
PATH_AUTOCLEANED      = PATH_PLACEHOLDER; 
PATH_WAVELETS         = PATH_PLACEHOLDER; 
PATH_TFDECOMP         = PATH_PLACEHOLDER; 
PATH_CLUSTSTATS       = PATH_PLACEHOLDER; 
PATH_PLOT             = PATH_PLACEHOLDER; 

% ======================= SUBJECTS ====================================================

% Define subjects (N=33)
subject_list = {'Exp20_0003', 'Exp20_0009', 'Exp20_0010', 'Exp20_0015', 'Exp20_0018',...
                'Exp20_0021', 'Exp20_0024', 'Exp20_0025', 'Exp20_0036', 'Exp20_0037',...
                'Exp20_0040', 'Exp20_0042', 'Exp20_0049', 'Exp20_0050', 'Exp20_0054',...
                'Exp20_0057', 'Exp20_0059', 'Exp20_0061', 'Exp20_0062', 'Exp20_0071',...
                'Exp20_0076', 'Exp20_0082', 'Exp20_0084', 'Exp20_0086', 'Exp20_0088',...
                'Exp20_0089', 'Exp20_0091', 'Exp20_0092', 'Exp20_0093', 'Exp20_0095',...
                'Exp20_0098', 'Exp20_0100', 'Exp20_0067'};

% This is a switch. Include all parts you want to be executed.
to_execute = {'part1'};

% ======================= PART1: CODIG AND PREPROCESSING ===============================

% Preprocessing
if ismember('part1', to_execute)

    % Init EEGlab
    addpath(PATH_EEGLAB);
	[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

	% Iterating subject list
	for s = 1 : length(subject_list)

		subject = subject_list{s};
        id = str2num(subject(end - 3 : end));
	    EEG =  pop_loadbv(PATH_RAW_DATA, [subject '.vhdr'], [], []);
	    EEG = pop_select(EEG, 'channel', [1 : 32]);
	    EEG.chanlocs_original = EEG.chanlocs;

	    % Define stimulus events
	    impevents = {'S 70', 'S 71', 'S 72', 'S 73', 'S 74', 'S 75', 'S 76', 'S 77'}; % Imperative stimuli
	    fcevents = {'S 50', 'S 51', 'S 52', 'S 53', 'S 54', 'S 55', 'S 56', 'S 57'}; % Fixation cross events
	    correvents = {'S100' 'S101' 'S102' 'S103' 'S104' 'S105' 'S106' 'S107'}; % Correct responses

	    % Find stim indices
	    stimidx = find(ismember({EEG.event.type}, impevents));

	    % New event struct
        new_events = struct('latency', {},...
                            'type', {},...
                            'code', {},... 
                            'accuracy', {},... 
                            'rt', {},...   
                            'condition', {},...
                            'task', {},...  
                            'switch', {},...      
                            'tut', {},...    
                            'position', {},...     
                            'eventid', {},...           
                            'urevent', {},...
                            'duration', {}...
                            );

        % Thhi is a counter
        ecount = 0;

	    % Detect TUT (tsak unrelated thoughts) event related stim events
	    trials = {[], [], []};
	    tuttype = [];
	    for e = 1 : length(stimidx)
	    	for f = stimidx(e) : stimidx(e) + 6 % Loop for TUT event 
	    		if f > length(EEG.event)
	    			break;
	    		end	
		    	if strcmp(EEG.event(f).type, 'S 82')
		    		trials{1}(end + 1) = stimidx(e);
        			tuttype(end + 1) = 0;
        			break;
        		elseif strcmp(EEG.event(f).type, 'S 81')
        			trials{1}(end + 1) = stimidx(e);
        			tuttype(end + 1) = 1;
        			break;
        		end
	    	end
	    end

	  	% Loop TUT events and detect t-1 and t-2 trials
	  	for e = 1 : length(trials{1})
	  		counter = 1;
	  		for f = trials{1}(e) - 1 : -1 : trials{1}(e) - 20
	  			if ismember(EEG.event(f).type, impevents)
	  				counter = counter + 1;
	  				if counter > 3
	  					break;
	  				end
	  				trials{counter}(end + 1) = f;
	  			end
	  		end
	  	end

	  	% Detect behavior
	  	rt = {[], [], []};
	  	acc = {[], [], []};
	  	cnd = {[], [], []};
	  	for t = 1 : 3
	  		for e = 1 : length(trials{t})

	  			% Get accuracy
	  			acc{t}(e) = 0;
	  			for f = trials{t}(e) : trials{t}(e) + 3
	  				if ismember(EEG.event(f).type, correvents)
	  					acc{t}(e) = 1;
	  					break;
	  				end
	  			end

	  			% Get rt
  				if acc{t}(e)
  					rt{t}(e) = EEG.event(f).latency - EEG.event(trials{t}(e)).latency;
  				else
  					rt{t}(e) = NaN;
  				end

				% Get condition
        		if ismember(EEG.event(trials{t}(e)).type, {'S 70', 'S 71'})
        			cnd{t}(e) = 1;
        		elseif ismember(EEG.event(trials{t}(e)).type, {'S 72', 'S 73'})
        			cnd{t}(e) = 2;
        		elseif ismember(EEG.event(trials{t}(e)).type, {'S 74', 'S 75'})
        			cnd{t}(e) = 3;
        		elseif ismember(EEG.event(trials{t}(e)).type, {'S 76', 'S 77'})
        			cnd{t}(e) = 4;
        		end
	  		end
	  	end

	  	% Loop for fixcrosses
	  	fc = {[], [], []};
	  	for t = 1 : 3
	  		for e = 1 : length(trials{t})
	  			for f = trials{t}(e) : -1 : trials{t}(e) - 4
	  				if ismember(EEG.event(f).type, fcevents)
	  					fc{t}(e) = f;
	  				end
	  			end
	  		end
	  	end

		% Create stim events
	  	for t = 1 : 3
	  		for e = 1 : length(trials{t})
        		ecount = ecount + 1;
                new_events(ecount).latency = EEG.event(trials{t}(e)).latency;
                new_events(ecount).type = ['stim' num2str(tuttype(e))];
                new_events(ecount).code = ['stim' num2str(tuttype(e))];
                new_events(ecount).accuracy = acc{t}(e);
                new_events(ecount).rt = rt{t}(e);
                new_events(ecount).condition = cnd{t}(e);
                if cnd{t}(e) <= 2
                    new_events(ecount).task = 'lessmore';
                else
                    new_events(ecount).task = 'oddeven';
                end
                if mod(cnd{t}(e), 2)
                    new_events(ecount).switch = 'repeat';
                else
                    new_events(ecount).switch = 'switch';
                end
                new_events(ecount).tut = tuttype(e);
                new_events(ecount).position = -(t - 1);
                new_events(ecount).eventid = t * 1000 + e;
                new_events(ecount).urevent = ecount;
                new_events(ecount).duration = 1;
            end 
    	end 

    	% Create fixcross events
	  	for t = 1 : 3
	  		for e = 1 : length(fc{t})
        		ecount = ecount + 1;
                new_events(ecount).latency = EEG.event(fc{t}(e)).latency;
                new_events(ecount).type = ['fix' num2str(tuttype(e))];
                new_events(ecount).code = ['fix' num2str(tuttype(e))];
                new_events(ecount).accuracy = acc{t}(e);
                new_events(ecount).rt = rt{t}(e);
                new_events(ecount).condition = cnd{t}(e);
                if cnd{t}(e) <= 2
                    new_events(ecount).task = 'lessmore';
                else
                    new_events(ecount).task = 'oddeven';
                end
                if mod(cnd{t}(e), 2)
                    new_events(ecount).switch = 'repeat';
                else
                    new_events(ecount).switch = 'switch';
                end
                new_events(ecount).tut = tuttype(e);
                new_events(ecount).position = -(t - 1);
                new_events(ecount).eventid = t * 1000 + e;
                new_events(ecount).urevent = ecount;
                new_events(ecount).duration = 1;
            end 
    	end 

	    % Add already existing boundaries
        for e = 1 : length(EEG.event)
            if strcmpi(EEG.event(e).type, 'boundary')
                ecount = ecount + 1;
                new_events(ecount).latency = EEG.event(e).latency;
                new_events(ecount).type = 'boundary';
                new_events(ecount).code = 'boundary';
                new_events(ecount).accuracy = NaN;
	            new_events(ecount).rt = NaN;
	            new_events(ecount).condition = NaN;
                new_events(ecount).task = NaN;
                new_events(ecount).switch = NaN;
	            new_events(ecount).tut = NaN;
	            new_events(ecount).position = NaN;
                new_events(ecount).eventid = NaN;
                new_events(ecount).urevent = ecount;
                new_events(ecount).duration = 1;
            end
        end

        % Replace events by new events
        EEG.event = new_events;
        EEG = eeg_checkset(EEG, 'eventconsistency');
 
        % Bandpass filter - function requires ERPlab extension (www.github.com/lucklab/erplab) 
        EEG = pop_basicfilter(EEG, [1 : EEG.nbchan], 'Cutoff', [1, 30], 'Design', 'butter', 'Filter', 'bandpass', 'Order', 4, 'RemoveDC', 'on', 'Boundary', 'boundary');
   
 	    % Bad channel detection
 	    [EEG, i1] = pop_rejchan(EEG, 'elec', [1 : EEG.nbchan], 'threshold', 10, 'norm', 'on', 'measure', 'kurt');
        [EEG, i2] = pop_rejchan(EEG, 'elec', [1 : EEG.nbchan], 'threshold', 5, 'norm', 'on', 'measure', 'prob');
        EEG.chans_rejected = horzcat(i1, i2);
 	    EEG.chans_rejected_n = length(horzcat(i1, i2));

        % Rereference to common average reference
        EEG = pop_reref(EEG, []);

        % Resample data
        EEG = pop_resample(EEG, 200);

        % Epoch data
        EEG = pop_epoch(EEG, {'fix0', 'fix1', 'stim0', 'stim1'}, [-2, 2], 'newname', [subject '_seg'], 'epochinfo', 'yes');
        EEG = pop_rmbase(EEG, [-500, -200]);
        
        % Automatically reject epochs before running ICA
        EEG.segs_original_n = size(EEG.data, 3);
        [EEG, rejsegs] = pop_autorej(EEG, 'nogui', 'on', 'threshold', 1000, 'startprob', 5, 'maxrej', 5, 'eegplot', 'off');
        EEG.segs_rejected_before_ica = length(rejsegs);

        % Run ICA
        EEG = pop_runica(EEG, 'extended', 1, 'interupt', 'on');

        % Save dataset with ICs
        EEG = pop_saveset(EEG, 'filename', [subject '_icset.set'], 'filepath', PATH_ICSET, 'check', 'on', 'savemode', 'twofiles');

    end % End subject loop

end % End part1

% ======================= PART2: IC-BASED DATA CLEANING ================================================

% Stuff
if ismember('part2', to_execute)

    % Init EEGlab
    addpath(PATH_EEGLAB);
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

    preprostats = [];

    % Iterating subject list
    for s = 1 : length(subject_list)

        subject = subject_list{s};
        id = str2num(subject(end - 3 : end));
        preprostats(s, 1) = id;

        % Load data
        EEG = pop_loadset('filename', [subject '_icset.set'], 'filepath', PATH_ICSET, 'loadmode', 'all');
        preprostats(s, 2) = EEG.chans_rejected_n;

        % Run IClabel
        EEG = iclabel(EEG);
        EEG.ICout_IClabel = find(EEG.etc.ic_classification.ICLabel.classifications(:, 1) < 0.5);

        % Remove components
        EEG = pop_subcomp(EEG, EEG.ICout_IClabel, 0);
        preprostats(s, 3) = length(EEG.ICout_IClabel);

        % Automated detection and removal of bad epochs
        [EEG, rejsegs] = pop_autorej(EEG, 'nogui', 'on', 'threshold', 1000, 'startprob', 5, 'maxrej', 5);
        EEG.segs_rejected_after_ica = length(rejsegs);
        EEG.segs_rejected_overall_percentage = ((EEG.segs_rejected_before_ica + EEG.segs_rejected_after_ica) / EEG.segs_original_n) * 100;
        preprostats(s, 4) = EEG.segs_rejected_overall_percentage;

        % Interpolate missing channels
        EEG = pop_interp(EEG, EEG.chanlocs_original, 'spherical');

        % Remove non-epoch-defining events
        EEG.event = EEG.event(mod(cell2mat({EEG.event.latency}), EEG.pnts) >= EEG.pnts / 2 - 5 & mod(cell2mat({EEG.event.latency}), EEG.pnts) <= EEG.pnts / 2 + 5);
        EEG = eeg_checkset(EEG, 'eventconsistency');

        % Exclude as well those events, for which the respective counterpart (fixation cross or imperative stimulus) has been already removed from the dataset 
        [C, ia, ic] = unique(cell2mat({EEG.event.eventid}));
        a_counts = accumarray(ic, 1);
        eventid_counts = [C', a_counts];
        tokeep = find(ismember(cell2mat({EEG.event.eventid}), eventid_counts(eventid_counts(:, 2) == 2, 1)));
        EEG = pop_select(EEG, 'trial', tokeep);

        % Get indices of trial types
        EEG.epoch_idx_labels = {'fix0_rep', 'fix1_rep', 'sti0_rep', 'sti1_rep', 'fix0_swi', 'fix1_swi', 'sti0_swi', 'sti1_swi'};
        EEG.epoch_idx{1} = cell2mat({EEG.event(find(strcmpi({EEG.event.switch}, 'repeat') & strcmpi({EEG.event.type},  'fix0'))).epoch}); 
        EEG.epoch_idx{2} = cell2mat({EEG.event(find(strcmpi({EEG.event.switch}, 'repeat') & strcmpi({EEG.event.type},  'fix1'))).epoch}); 
        EEG.epoch_idx{3} = cell2mat({EEG.event(find(strcmpi({EEG.event.switch}, 'repeat') & strcmpi({EEG.event.type}, 'stim0'))).epoch});
        EEG.epoch_idx{4} = cell2mat({EEG.event(find(strcmpi({EEG.event.switch}, 'repeat') & strcmpi({EEG.event.type}, 'stim1'))).epoch});
        EEG.epoch_idx{5} = cell2mat({EEG.event(find(strcmpi({EEG.event.switch}, 'switch') & strcmpi({EEG.event.type},  'fix0'))).epoch}); 
        EEG.epoch_idx{6} = cell2mat({EEG.event(find(strcmpi({EEG.event.switch}, 'switch') & strcmpi({EEG.event.type},  'fix1'))).epoch}); 
        EEG.epoch_idx{7} = cell2mat({EEG.event(find(strcmpi({EEG.event.switch}, 'switch') & strcmpi({EEG.event.type}, 'stim0'))).epoch});
        EEG.epoch_idx{8} = cell2mat({EEG.event(find(strcmpi({EEG.event.switch}, 'switch') & strcmpi({EEG.event.type}, 'stim1'))).epoch});
        for c = 1 : 8
            EEG.epoch_idx_n(c) = length(EEG.epoch_idx{c});
            preprostats(s, 4 + c) = length(EEG.epoch_idx{c});
        end

        % Save data
        EEG = pop_saveset(EEG, 'filename', [subject '_autocleaned.set'], 'filepath', PATH_AUTOCLEANED, 'check', 'on', 'savemode', 'twofiles');

    end % End subject loop

    % Save table with preprocessing statistics
    dlmwrite([PATH_AUTOCLEANED 'preprostats.csv'], preprostats);

    % Print some statistics for excluded ICs
    fprintf('\nRejected ICs: mean %i, stdev %i\n', mean(preprostats(:, 3)), std(preprostats(:, 3)));

end % End part2

% At this point, a visual inspection of the ICs was conducted. There were some remaining artifact ICs, which were removed (most of these ICs represented mostly ECG activity).

% ==== PART3: HISTOGRAMS FOR THE TEMPORAL DISTRIBUTION OF TUT EVENTS & REJECTED IC POSITION =========

% Stuff
if ismember('part3', to_execute)

    % Init EEGlab
    addpath(PATH_EEGLAB);
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

    % Iterating subject list
    y = [];
    icdist = [];
    for s = 1 : length(subject_list)

        subject = subject_list{s};
        id = str2num(subject(end - 3 : end));

        % Load data
        EEG = pop_loadset('filename', [subject '_autocleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

        % Get tut latencies
        x = cell2mat({EEG.event(find(strcmpi({EEG.event.type}, 'stim1') & cell2mat({EEG.event.position}) == 0)).latency});
        y = [y, x];

        % Get rejected IC positions
        x = EEG.ICout_IClabel';
        icdist = [icdist, x];
        
    end % End subject loop

    % Scale
    y = y / max(y);
    x = [1 : length(y)] / length(y);

    % Compute correlation of time on task and numbre of reported task unrelated thoughts
    corrcoef(x, y)

    % Save histogram data
    z = hist(y, 30);
    dlmwrite([PATH_PLOT 'hist_tot_dat.csv'], z);
    z = hist(icdist, 32);
    dlmwrite([PATH_PLOT 'hist_ICout.csv'], z);


end % End part3

% ======================= PART4: TIME FREQUENCY DECOMPOSITION ===================================================================================================

% Preprocessing
if ismember('part4', to_execute)

    % Init ft
    rmpath(PATH_EEGLAB);
    addpath(PATH_FIELDTRIP);
    ft_defaults;

    % Set complex Morlet wavelet parameters
    n_frq = 50;
    frqrange = [2, 30];
    tfres_range = [600, 50];
    data.hdr = ft_read_header([PATH_AUTOCLEANED subject_list{1} '_autocleaned.set']);

	% Set wavelet time
    wtime = -2 : 1 / data.hdr.Fs : 2;
    
    % Determine fft frqs
	hz = linspace(0, data.hdr.Fs, length(wtime));

    % Create wavelet frequencies and tapering Gaussian widths in temporal domain
    tf_freqs = logspace(log10(frqrange(1)), log10(frqrange(2)), n_frq);
    fwhmTs = logspace(log10(tfres_range(1)), log10(tfres_range(2)), n_frq);

	% Init matrices for wavelets
	cmw = zeros(length(tf_freqs), length(wtime));
	cmwX = zeros(length(tf_freqs), length(wtime));
    tlim = zeros(1, length(tf_freqs));
    
    % These will contain the wavelet widths as full width at 
    % half maximum in the temporal and spectral domain
	obs_fwhmT = zeros(1, length(tf_freqs));
	obs_fwhmF = zeros(1, length(tf_freqs));

	% Create the wavelets
	for frq = 1 : length(tf_freqs)

		% Create wavelet with tapering gaussian corresponding to desired width in temporal domain
		cmw(frq, :) = exp(2 * 1i * pi * tf_freqs(frq) .* wtime) .* exp((-4 * log(2) * wtime.^2) ./ (fwhmTs(frq) / 1000)^2);

		% Normalize wavelet
		cmw(frq, :) = cmw(frq, :) ./ max(cmw(frq, :));

		% Create normalized freq domain wavelet
		cmwX(frq, :) = fft(cmw(frq, :)) ./ max(fft(cmw(frq, :)));

		% Determine observed fwhmT
		midt = dsearchn(wtime', 0);
		cmw_amp = abs(cmw(frq, :)) ./ max(abs(cmw(frq, :))); % Normalize cmw amplitude
		obs_fwhmT(frq) = wtime(midt - 1 + dsearchn(cmw_amp(midt : end)', 0.5)) - wtime(dsearchn(cmw_amp(1 : midt)', 0.5));

		% Determine observed fwhmF
		idx = dsearchn(hz', tf_freqs(frq));
		cmwx_amp = abs(cmwX(frq, :)); 
		obs_fwhmF(frq) = hz(idx - 1 + dsearchn(cmwx_amp(idx : end)', 0.5) - dsearchn(cmwx_amp(1 : idx)', 0.5));

	end

    % Iterating subject list
    for s = 1 : length(subject_list)

        % Current subject
        subject = subject_list{s};

        % Construct a a ft_datatype_raw struct (a fieldtrip datatype in order to prepare for fieldtrips statistics) 
        data = [];
        data.hdr = ft_read_header([PATH_AUTOCLEANED subject '_autocleaned.set']);
        d = ft_read_data([PATH_AUTOCLEANED subject '_autocleaned.set'], 'header', data.hdr);
        for t = 1 : size(d, 3)
            data.trial{t} = squeeze(d(:, :, t));
            data.time{t} = data.hdr.orig.times / 1000;
            data.sampleinfo(t, :) = [data.hdr.orig.event(t).latency - 400, data.hdr.orig.event(t).latency + 399];
        end
        data.label = data.hdr.label;
        data.fsample = data.hdr.Fs;
        for c = 1 : 8
            data.trialinfo(data.hdr.orig.epoch_idx{c}, 1) = c;
        end
        data.trialinfo(:, 2) = 0;
        data.trialinfo(ismember(data.trialinfo(:, 1), [3, 4, 7, 8]), 2) = 1; % fix=0, stim=1
        data.trialinfo(:, 3) = 0;
        data.trialinfo(ismember(data.trialinfo(:, 1), [2, 4, 6, 8]), 3) = 1; % tut0=0, tut1=1
        data.trialinfo(:, 4) = 0;
        data.trialinfo(ismember(data.trialinfo(:, 1), [5, 6, 7, 8]), 4) = 1; % stay=0, switch=1
        data.trialinfo(:, 5) = cell2mat({data.hdr.orig.event.rt});
        data.trialinfo(:, 6) = cell2mat({data.hdr.orig.event.accuracy});
        data.trialinfo(:, 7) = cell2mat({data.hdr.orig.event.eventid});
        data.trialinfo(:, 8) = cell2mat({data.hdr.orig.event.position});

        % Exclude bad rt trials
        cutoff_hi_sd = 3;
        cutoff_lo_ms = 150;
        todrop = find(data.trialinfo(:, 5) > median(data.trialinfo(:, 5)) + std(data.trialinfo(:, 5)) * cutoff_hi_sd | data.trialinfo(:, 5) < cutoff_lo_ms);
        data.trialinfo(todrop, :) = [];
        d(:, :, todrop) = [];

        % Time frequency decomposition
        for ch = 1 : size(d, 1)

            % Talk
            fprintf('\ntf decomp subject %i/%i | chan %i/%i...\n', s, numel(subject_list), ch, size(d, 1));

            % Pick channel data
            dch = squeeze(d(ch, :, :));

            % Set convolution length
            convlen = size(dch, 1) * size(dch, 2) + size(cmw, 2) - 1;

            % Transform cmw to freqency domain and scale
            cmwX = zeros(n_frq, convlen);
            for f = 1 : n_frq
                cmwX(f, :) = fft(cmw(f, :), convlen);
                cmwX(f, :) = cmwX(f, :) ./ max(cmwX(f, :));
            end

            % Get TF-power
            powcube = NaN(n_frq, size(dch, 1), size(dch, 2));
            tmp = fft(reshape(double(dch), 1, []), convlen);
            for f = 1 : n_frq
                as = ifft(cmwX(f, :) .* tmp); 
                as = as(((size(cmw, 2) - 1) / 2) + 1 : end - ((size(cmw, 2) - 1) / 2));
                as = reshape(as, size(dch, 1), size(dch, 2));
                powcube(f, :, :) = abs(as) .^ 2;          
            end

            % Cut edge artifacts
            pruned_segs = [-1700, 1700];
            tf_times = data.hdr.orig.times(dsearchn(data.hdr.orig.times', pruned_segs(1)) : dsearchn(data.hdr.orig.times', pruned_segs(2)));
            powcube = powcube(:, dsearchn(data.hdr.orig.times', pruned_segs(1)) : dsearchn(data.hdr.orig.times', pruned_segs(2)), :);

            % Save single trial power and metadata
            save([PATH_TFDECOMP 'powcube_latencies'], 'tf_times');
            save([PATH_TFDECOMP 'powcube_freqs'], 'tf_freqs');
            save([PATH_TFDECOMP 'powcube_' subject '_chan_' num2str(ch)], 'powcube');

            % Calc ersp baselines
            ersp_bl = [-500, -200];
            tmp = squeeze(mean(powcube(:, :, data.trialinfo(:, 2) == 0), 3));
            [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
            [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
            blvals_fix = squeeze(mean(tmp(:, blidx1 : blidx2), 2));
            ersp_bl = [-500, -200];
            tmp = squeeze(mean(powcube(:, :, data.trialinfo(:, 2) == 1), 3));
            [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
            [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
            blvals_sti = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

            % Prune fixcross locked data
            pruned_segs = [-1000, 1500];
            time_idx_fix = dsearchn(tf_times', pruned_segs(1)) : dsearchn(tf_times', pruned_segs(2));
            tf_times_fix = tf_times(time_idx_fix);
            powcube_fix0 = powcube(:, time_idx_fix, data.trialinfo(:, 2) == 0 & data.trialinfo(:, 3) == 0);
            powcube_fix1 = powcube(:, time_idx_fix, data.trialinfo(:, 2) == 0 & data.trialinfo(:, 3) == 1);

            % Prune stimlocked data
            pruned_segs = [-500, 1500];
            time_idx_sti = dsearchn(tf_times', pruned_segs(1)) : dsearchn(tf_times', pruned_segs(2));
            tf_times_sti = tf_times(time_idx_sti);
            powcube_sti0_stay = powcube(:, time_idx_sti, data.trialinfo(:, 2) == 1 & data.trialinfo(:, 3) == 0 & data.trialinfo(:, 4) == 0);
            powcube_sti1_stay = powcube(:, time_idx_sti, data.trialinfo(:, 2) == 1 & data.trialinfo(:, 3) == 1 & data.trialinfo(:, 4) == 0);
            powcube_sti0_switch = powcube(:, time_idx_sti, data.trialinfo(:, 2) == 1 & data.trialinfo(:, 3) == 0 & data.trialinfo(:, 4) == 1);
            powcube_sti1_switch = powcube(:, time_idx_sti, data.trialinfo(:, 2) == 1 & data.trialinfo(:, 3) == 1 & data.trialinfo(:, 4) == 1);
            powcube_sti0 = powcube(:, time_idx_sti, data.trialinfo(:, 2) == 1 & data.trialinfo(:, 3) == 0);
            powcube_sti1 = powcube(:, time_idx_sti, data.trialinfo(:, 2) == 1 & data.trialinfo(:, 3) == 1);
            powcube_stay = powcube(:, time_idx_sti, data.trialinfo(:, 2) == 1 & data.trialinfo(:, 4) == 0);
            powcube_switch = powcube(:, time_idx_sti, data.trialinfo(:, 2) == 1 & data.trialinfo(:, 4) == 1);

            % Calc ersps for conditions and create a struct according to ft_datatype_freq
            dtf_fix0.powspctrm(ch, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_fix0, 3), blvals_fix));
            dtf_fix1.powspctrm(ch, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_fix1, 3), blvals_fix));
            dtf_sti0_stay.powspctrm(ch, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_sti0_stay, 3), blvals_sti));
            dtf_sti1_stay.powspctrm(ch, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_sti1_stay, 3), blvals_sti));
            dtf_sti0_switch.powspctrm(ch, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_sti0_switch, 3), blvals_sti));
            dtf_sti1_switch.powspctrm(ch, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_sti1_switch, 3), blvals_sti));
            dtf_sti0.powspctrm(ch, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_sti0, 3), blvals_sti));
            dtf_sti1.powspctrm(ch, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_sti1, 3), blvals_sti));
            dtf_stay.powspctrm(ch, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_stay, 3), blvals_sti));
            dtf_switch.powspctrm(ch, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_switch, 3), blvals_sti));
    
        end % End chanit

        % Add metadata to ft_datatype_freq
        dtf_fix0.dimord = 'chan_freq_time';
        dtf_fix0.label = data.hdr.label;
        dtf_fix0.freq = tf_freqs;
        dtf_fix0.time = tf_times_fix;
        dtf_fix1.dimord = 'chan_freq_time';
        dtf_fix1.label = data.hdr.label;
        dtf_fix1.freq = tf_freqs;
        dtf_fix1.time = tf_times_fix;
        dtf_sti0_stay.dimord = 'chan_freq_time';
        dtf_sti0_stay.label = data.hdr.label;
        dtf_sti0_stay.freq = tf_freqs;
        dtf_sti0_stay.time = tf_times_sti;
        dtf_sti0_switch.dimord = 'chan_freq_time';
        dtf_sti0_switch.label = data.hdr.label;
        dtf_sti0_switch.freq = tf_freqs;
        dtf_sti0_switch.time = tf_times_sti;
        dtf_sti1_stay.dimord = 'chan_freq_time';
        dtf_sti1_stay.label = data.hdr.label;
        dtf_sti1_stay.freq = tf_freqs;
        dtf_sti1_stay.time = tf_times_sti;
        dtf_sti1_switch.dimord = 'chan_freq_time';
        dtf_sti1_switch.label = data.hdr.label;
        dtf_sti1_switch.freq = tf_freqs;
        dtf_sti1_switch.time = tf_times_sti;
        dtf_sti0.dimord = 'chan_freq_time';
        dtf_sti0.label = data.hdr.label;
        dtf_sti0.freq = tf_freqs;
        dtf_sti0.time = tf_times_sti;
        dtf_sti1.dimord = 'chan_freq_time';
        dtf_sti1.label = data.hdr.label;
        dtf_sti1.freq = tf_freqs;
        dtf_sti1.time = tf_times_sti;
        dtf_stay.dimord = 'chan_freq_time';
        dtf_stay.label = data.hdr.label;
        dtf_stay.freq = tf_freqs;
        dtf_stay.time = tf_times_sti;
        dtf_switch.dimord = 'chan_freq_time';
        dtf_switch.label = data.hdr.label;
        dtf_switch.freq = tf_freqs;
        dtf_switch.time = tf_times_sti;
  
        % Save averages as ft struct
        save([PATH_TFDECOMP 'tfdat_fix0_' subject], 'dtf_fix0');
        save([PATH_TFDECOMP 'tfdat_fix1_' subject], 'dtf_fix1');
        save([PATH_TFDECOMP 'tfdat_sti0_stay_' subject], 'dtf_sti0_stay');
        save([PATH_TFDECOMP 'tfdat_sti1_stay_' subject], 'dtf_sti1_stay');
        save([PATH_TFDECOMP 'tfdat_sti0_switch_' subject], 'dtf_sti0_switch');
        save([PATH_TFDECOMP 'tfdat_sti1_switch_' subject], 'dtf_sti1_switch');
        save([PATH_TFDECOMP 'tfdat_sti0_' subject], 'dtf_sti0');
        save([PATH_TFDECOMP 'tfdat_sti1_' subject], 'dtf_sti1');
        save([PATH_TFDECOMP 'tfdat_stay_' subject], 'dtf_stay');
        save([PATH_TFDECOMP 'tfdat_switch_' subject], 'dtf_switch');

        % Save trialinfo
        trialinfo = data.trialinfo;
        save([PATH_TFDECOMP 'trialinfo_' subject], 'trialinfo');

    end % End subject loop

end % End part4

% ============== PART5: DEMONSTRATE EFFECTS OF BASELINE CHOICE ==========================

% Preprocessing
if ismember('part5', to_execute)

    % Init EEGlab
    addpath(PATH_EEGLAB);
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

    % Select electrode (10 is a right parietal electrode...)
    elec = 10;

    % Load time frequency parametres
    load([PATH_TFDECOMP 'powcube_latencies']);
    load([PATH_TFDECOMP 'powcube_freqs']);

    % Get time indices for pruned segmets
    pruned_segs = [-500, 1000];
    time_idx = dsearchn(tf_times', pruned_segs(1)) : dsearchn(tf_times', pruned_segs(2));
    prunetime = tf_times(time_idx);

    % Init data matrix
    outdat = zeros(length(subject_list), 3, 4, length(tf_freqs), length(prunetime));
    
    % Iterating subject list
    for s = 1 : length(subject_list)

        % Current subject
        subject = subject_list{s};

        % Save single trial power (powcube) of chanel 11 (posterior)
        load([PATH_TFDECOMP 'powcube_' subject '_chan_' num2str(elec)]);

        % Load trialinfo 
        load([PATH_TFDECOMP 'trialinfo_' subject]');

        % Get indices of baseline
        ersp_bl = [-500, -200];
        [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        [~, blidx2] = min(abs(tf_times - ersp_bl(2)));

        % Calculate ersp baselines for fixation-cross locked data across all conditions
        tmp = squeeze(mean(powcube(:, :, trialinfo(:, 2) == 0), 3));
        blvals_fix_all = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

        % Calculate ersp baselines for stimulus locked data across all conditions
        tmp = squeeze(mean(powcube(:, :, trialinfo(:, 2) == 1), 3));
        blvals_sti_all = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

        % Calculate ersp baselines for fixation-cross locked data for on task trials
        tmp = squeeze(mean(powcube(:, :, trialinfo(:, 2) == 0 & trialinfo(:, 3) == 0), 3));
        blvals_fix_ot = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

        % Calculate ersp baselines for fixation-cross locked data for mind wandering trials
        tmp = squeeze(mean(powcube(:, :, trialinfo(:, 2) == 0 & trialinfo(:, 3) == 1), 3));
        blvals_fix_mw = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

        % Calculate ersp baselines for stimulus locked data for on task trials
        tmp = squeeze(mean(powcube(:, :, trialinfo(:, 2) == 1 & trialinfo(:, 3) == 0), 3));
        blvals_sti_ot = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

        % Calculate ersp baselines for stimulus locked data for mind wandering trials
        tmp = squeeze(mean(powcube(:, :, trialinfo(:, 2) == 1 & trialinfo(:, 3) == 1), 3));
        blvals_sti_mw = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

        % Prune data
        powcube_fix0 = powcube(:, time_idx, trialinfo(:, 2) == 0 & trialinfo(:, 3) == 0);
        powcube_fix1 = powcube(:, time_idx, trialinfo(:, 2) == 0 & trialinfo(:, 3) == 1);
        powcube_sti0 = powcube(:, time_idx, trialinfo(:, 2) == 1 & trialinfo(:, 3) == 0);
        powcube_sti1 = powcube(:, time_idx, trialinfo(:, 2) == 1 & trialinfo(:, 3) == 1);

        % Apply baseline normalization with common baseline
        outdat(s, 1, 1, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_fix0, 3), blvals_fix_all));
        outdat(s, 1, 2, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_sti0, 3), blvals_sti_all));
        outdat(s, 1, 3, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_fix1, 3), blvals_fix_all)); 
        outdat(s, 1, 4, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_sti1, 3), blvals_sti_all));
       
        % Apply condition specific baseline
        outdat(s, 2, 1, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_fix0, 3), blvals_fix_ot));
        outdat(s, 2, 2, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_sti0, 3), blvals_sti_ot));
        outdat(s, 2, 3, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_fix1, 3), blvals_fix_mw));
        outdat(s, 2, 4, :, :) = 10 * log10(bsxfun(@rdivide, mean(powcube_sti1, 3), blvals_sti_mw));

        % No baseline at all
        outdat(s, 3, 1, :, :) = mean(powcube_fix0, 3);
        outdat(s, 3, 2, :, :) = mean(powcube_sti0, 3);
        outdat(s, 3, 3, :, :) = mean(powcube_fix1, 3);
        outdat(s, 3, 4, :, :) = mean(powcube_sti1, 3);

    end % End subject loop

    % Iterate baselines and segments, save 2d data and collect alpha traces
    alpha_traces = zeros(12, length(prunetime));
    bl_labs = {'all', 'cnd', 'non'};
    seg_labs = {'fix0', 'sti0', 'fix1', 'sti1'};
    cnt = 0;
    for bl = 1 : 3
        for seg = 1 : 4

            % Get relevant data and average across subjects
            tmp = squeeze(mean(outdat(:, bl, seg, :, :), 1));

            % Save 2d data
            dlmwrite([PATH_PLOT, '/bldemo/pd_' bl_labs{bl} '_' seg_labs{seg} '.csv'], tmp);

            % Calculate alpha trace
            cnt = cnt + 1;
            alpha_traces(cnt, :) = mean(tmp(tf_freqs >= 8 & tf_freqs <= 12, :), 1); 
        end
    end

    % Save alpha traces
    dlmwrite([PATH_PLOT, '/bldemo/pd_alpha_traces.csv'], alpha_traces);

    % Save time-frequency parameters
    dlmwrite([PATH_PLOT, '/bldemo/frqs.csv'], tf_freqs);
    dlmwrite([PATH_PLOT, '/bldemo/time.csv'], prunetime);

end % End part5

% ======================= PART6: CALCULATE CLUSTER PERMUTATION STATISTIC ===================================================================================================

% Preprocessing
if ismember('part6', to_execute)

    % Init ft
    addpath(PATH_FIELDTRIP);
    ft_defaults;

    % Load power data
    for s = 1 : length(subject_list)
        subject = subject_list{s};
        load([PATH_TFDECOMP 'tfdat_fix0_' subject]);
        load([PATH_TFDECOMP 'tfdat_fix1_' subject]);
        load([PATH_TFDECOMP 'tfdat_sti0_stay_' subject]);
        load([PATH_TFDECOMP 'tfdat_sti1_stay_' subject]);
        load([PATH_TFDECOMP 'tfdat_sti0_switch_' subject]);
        load([PATH_TFDECOMP 'tfdat_sti1_switch_' subject]);
        load([PATH_TFDECOMP 'tfdat_sti0_' subject]);
        load([PATH_TFDECOMP 'tfdat_sti1_' subject]);
        load([PATH_TFDECOMP 'tfdat_stay_' subject]);
        load([PATH_TFDECOMP 'tfdat_switch_' subject]);
        D_fix0{s} = dtf_fix0;
        D_fix1{s} = dtf_fix1;
        D_sti0_stay{s} = dtf_sti0_stay;
        D_sti1_stay{s} = dtf_sti1_stay;
        D_sti0_switch{s} = dtf_sti0_switch;
        D_sti1_switch{s} = dtf_sti1_switch;
        D_sti0{s} = dtf_sti0;
        D_sti1{s} = dtf_sti1;
        D_stay{s} = dtf_stay;
        D_switch{s} = dtf_switch;
    end

    % Calc power GA
    cfg=[];
    cfg.keepindividual = 'yes';
    GA_pow_fix0 = ft_freqgrandaverage(cfg, D_fix0{1, :});
    GA_pow_fix1 = ft_freqgrandaverage(cfg, D_fix1{1, :});
    GA_pow_sti0_stay = ft_freqgrandaverage(cfg, D_sti0_stay{1, :});
    GA_pow_sti1_stay = ft_freqgrandaverage(cfg, D_sti1_stay{1, :});
    GA_pow_sti0_switch = ft_freqgrandaverage(cfg, D_sti0_switch{1, :});
    GA_pow_sti1_switch = ft_freqgrandaverage(cfg, D_sti1_switch{1, :});
    GA_pow_sti0 = ft_freqgrandaverage(cfg, D_sti0{1, :});
    GA_pow_sti1 = ft_freqgrandaverage(cfg, D_sti1{1, :});
    GA_pow_stay = ft_freqgrandaverage(cfg, D_stay{1, :});
    GA_pow_switch = ft_freqgrandaverage(cfg, D_switch{1, :});

    % Calculate rep-swi diffs for both tut-conditions 
    GA_pow_diff0 = GA_pow_sti0_stay;
    GA_pow_diff1 = GA_pow_sti0_stay;
    GA_pow_diff0.powspctrm = GA_pow_sti0_stay.powspctrm - GA_pow_sti0_switch.powspctrm;
    GA_pow_diff1.powspctrm = GA_pow_sti1_stay.powspctrm - GA_pow_sti1_switch.powspctrm;

    % Define neighbours
    cfg                 = [];
    cfg.layout          = 'easycapM7.mat';
    cfg.feedback        = 'no';
    cfg.method          = 'triangulation'; 
    cfg.neighbours      = ft_prepare_neighbours(cfg, GA_pow_fix0);
    neighbours = cfg.neighbours;

    % Testparams
    testalpha  = 0.025; % Two sided!
    voxelalpha  = 0.01;
    nperm = 1000;

    % Set config. Same for all tests
    cfg = [];
    cfg.tail             = 0; % Two sided test!
    cfg.statistic        = 'depsamplesT';
    cfg.alpha            = testalpha;
    cfg.neighbours       = neighbours;
    cfg.minnbchan        = 2;
    cfg.method           = 'montecarlo';
    cfg.correctm         = 'cluster';
    cfg.clustertail      = 0;
    cfg.clusteralpha     = voxelalpha;
    cfg.clusterstatistic = 'maxsum';
    cfg.numrandomization = nperm;
    cfg.computecritval   = 'yes'; 
    cfg.ivar             = 1;
    cfg.uvar             = 2;
    cfg.design           = [ones(1, numel(subject_list)), 2 * ones(1, numel(subject_list)); 1 : numel(subject_list), 1 : numel(subject_list)];

    % The tests
    [stat_pow_fix] = ft_freqstatistics(cfg, GA_pow_fix0, GA_pow_fix1);  
    [stat_pow_sti_tut] = ft_freqstatistics(cfg, GA_pow_sti0, GA_pow_sti1);   
    [stat_pow_sti_seq] = ft_freqstatistics(cfg, GA_pow_stay, GA_pow_switch); 
    [stat_pow_sti_int] = ft_freqstatistics(cfg, GA_pow_diff0, GA_pow_diff1);

    % Save stats and grand averages
    save([PATH_CLUSTSTATS 'stat_pow_fix'], 'stat_pow_fix');
    save([PATH_CLUSTSTATS 'stat_pow_sti_tut'], 'stat_pow_sti_tut');
    save([PATH_CLUSTSTATS 'stat_pow_sti_seq'], 'stat_pow_sti_seq');
    save([PATH_CLUSTSTATS 'stat_pow_sti_int'], 'stat_pow_sti_int');
    save([PATH_CLUSTSTATS 'GA_pow_fix0'], 'GA_pow_fix0');
    save([PATH_CLUSTSTATS 'GA_pow_fix1'], 'GA_pow_fix1');
    save([PATH_CLUSTSTATS 'GA_pow_sti0_stay'], 'GA_pow_sti0_stay');
    save([PATH_CLUSTSTATS 'GA_pow_sti1_stay'], 'GA_pow_sti1_stay');
    save([PATH_CLUSTSTATS 'GA_pow_sti0_switch'], 'GA_pow_sti0_switch');
    save([PATH_CLUSTSTATS 'GA_pow_sti1_switch'], 'GA_pow_sti1_switch');
    save([PATH_CLUSTSTATS 'GA_pow_sti0'], 'GA_pow_sti0');
    save([PATH_CLUSTSTATS 'GA_pow_sti1'], 'GA_pow_sti1');
    save([PATH_CLUSTSTATS 'GA_pow_stay'], 'GA_pow_stay');
    save([PATH_CLUSTSTATS 'GA_pow_switch'], 'GA_pow_switch');

end % End part6

% ======================= PART7: VISUALIZE CLUSTER ===================================================================================================

% Preprocessing
if ismember('part7', to_execute)

    % Get chanlocs
    addpath(PATH_EEGLAB);
    eeglab;
    EEG = pop_loadset('filename', [subject_list{1} '_autocleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');
    chanlocs = EEG.chanlocs;

    % Load cluststats
    load([PATH_CLUSTSTATS 'stat_pow_fix']);
    load([PATH_CLUSTSTATS 'stat_pow_sti_tut']);
    load([PATH_CLUSTSTATS 'stat_pow_sti_seq']);
    load([PATH_CLUSTSTATS 'stat_pow_sti_int']);
    load([PATH_CLUSTSTATS 'GA_pow_fix0']);
    load([PATH_CLUSTSTATS 'GA_pow_fix1']);
    load([PATH_CLUSTSTATS 'GA_pow_sti0_stay']);
    load([PATH_CLUSTSTATS 'GA_pow_sti1_stay']);
    load([PATH_CLUSTSTATS 'GA_pow_sti0_switch']);
    load([PATH_CLUSTSTATS 'GA_pow_sti1_switch']);
    load([PATH_CLUSTSTATS 'GA_pow_sti0']);
    load([PATH_CLUSTSTATS 'GA_pow_sti1']);
    load([PATH_CLUSTSTATS 'GA_pow_stay']);
    load([PATH_CLUSTSTATS 'GA_pow_switch']);

    % Repeat testalpha
    testalpha  = 0.025;

    % Set colors
    cmap = 'jet';
    clinecol = 'k';

    % Identify significant clusters
    clusts = struct();
    cnt = 0;
    stat_names = {'stat_pow_fix', 'stat_pow_sti_tut', 'stat_pow_sti_seq', 'stat_pow_sti_int'};
    for s = 1 : numel(stat_names)
        stat = eval(stat_names{s});
        if ~isempty(stat.negclusters)
            neg_idx = find([stat.negclusters(1, :).prob] < testalpha);
            for c = 1 : numel(neg_idx)
                cnt = cnt + 1;
                clusts(cnt).testlabel = stat_names{s};
                clusts(cnt).clustnum = cnt;
                clusts(cnt).time = stat.time;
                clusts(cnt).freq = stat.freq;
                clusts(cnt).polarity = -1;
                clusts(cnt).prob = stat.negclusters(1, neg_idx(c)).prob;
                clusts(cnt).idx = stat.negclusterslabelmat == neg_idx(c);
                clusts(cnt).stats = clusts(cnt).idx .* stat.stat * -1;
                clusts(cnt).chans_sig = find(logical(mean(clusts(cnt).idx, [2,3])));
            end
        end
        if ~isempty(stat.posclusters)
            pos_idx = find([stat.posclusters(1, :).prob] < testalpha);
            for c = 1 : numel(pos_idx)
                cnt = cnt + 1;
                clusts(cnt).testlabel = stat_names{s};
                clusts(cnt).clustnum = cnt;
                clusts(cnt).time = stat.time;
                clusts(cnt).freq = stat.freq;
                clusts(cnt).polarity = 1;
                clusts(cnt).prob = stat.posclusters(1, pos_idx(c)).prob;
                clusts(cnt).idx = stat.posclusterslabelmat == pos_idx(c);
                clusts(cnt).stats = clusts(cnt).idx .* stat.stat;
                clusts(cnt).chans_sig = find(logical(mean(clusts(cnt).idx, [2,3])));
            end
        end
    end

    % Save cluster struct
    save([PATH_CLUSTSTATS 'significant_clusters.mat'], 'clusts');

    % Plot identified cluster
    for cnt = 1 : numel(clusts)
        figure('Visible', 'off'); clf;
        subplot(2, 2, 1)
        pd = squeeze(sum(clusts(cnt).stats, 1));
        contourf(clusts(cnt).time, clusts(cnt).freq, pd, 40, 'linecolor','none')
        hold on
        contour(clusts(cnt).time, clusts(cnt).freq, logical(squeeze(mean(clusts(cnt).idx, 1))), 1, 'linecolor', clinecol, 'LineWidth', 2)
        colormap(cmap)
        set(gca, 'xlim', [clusts(cnt).time(1), clusts(cnt).time(end)], 'clim', [-max(abs(pd(:))), max(abs(pd(:)))], 'YScale', 'log', 'YTick', [4, 8, 12, 20, 30])
        colorbar;
        title(['sum t across chans, plrt: ' num2str(clusts(cnt).polarity)], 'FontSize', 10)
        subplot(2, 2, 2)
        pd = squeeze(mean(clusts(cnt).idx, 1));
        contourf(clusts(cnt).time, clusts(cnt).freq, pd, 40, 'linecolor','none')
        hold on
        contour(clusts(cnt).time, clusts(cnt).freq, logical(squeeze(mean(clusts(cnt).idx, 1))), 1, 'linecolor', clinecol, 'LineWidth', 2)
        colormap(cmap)
        set(gca, 'xlim', [clusts(cnt).time(1), clusts(cnt).time(end)], 'clim', [-1, 1], 'YScale', 'log', 'YTick', [4, 8, 12, 20, 30])
        colorbar;
        title(['proportion chans significant'], 'FontSize', 10)
        subplot(2, 2, 3)
        pd = squeeze(sum(clusts(cnt).stats, [2, 3]));
        topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
        colormap(cmap)
        set(gca, 'clim', [-max(abs(pd(:))), max(abs(pd(:)))])
        colorbar;
        title(['sum t per electrode'], 'FontSize', 10)
        subplot(2, 2, 4)
        pd = squeeze(mean(clusts(cnt).idx, [2, 3]));
        topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
        colormap(cmap)
        set(gca, 'clim', [-1, 1])
        colorbar;
        title(['proportion tf-points significant'], 'FontSize', 10)
        saveas(gcf, [PATH_PLOT 'clustnum_' num2str(clusts(cnt).clustnum) '_' clusts(cnt).testlabel '.png']); 
    end

    % Inspect output and go on. From this point on informed analysis, i.e. it is known how many clusters have been identified.

    % Aggregate contour
    pd_tut_fix_contour = logical(squeeze(mean(clusts(1).idx, 1))) + logical(squeeze(mean(clusts(2).idx, 1))) ;
    pd_tut_sti_contour = logical(squeeze(mean(clusts(3).idx, 1))) + logical(squeeze(mean(clusts(4).idx, 1)));
    pd_seq_sti_contour = logical(squeeze(mean(clusts(5).idx, 1)));

    % Calculate effect sizes
    for ch = 1 : 32

        petasq = (squeeze(stat_pow_fix.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_pow_fix.stat(ch, :, :)) .^ 2) + (numel(subject_list) - 1));
        adj_petasq = petasq - (1 - petasq) .* (1 / (numel(subject_list) - 1));
        adjpetasq_tut_fix(ch, :, :) = adj_petasq;

        petasq = (squeeze(stat_pow_sti_tut.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_pow_sti_tut.stat(ch, :, :)) .^ 2) + (numel(subject_list) - 1));
        adj_petasq = petasq - (1 - petasq) .* (1 / (numel(subject_list) - 1));
        adjpetasq_tut_sti(ch, :, :) = adj_petasq;

        petasq = (squeeze(stat_pow_sti_seq.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_pow_sti_seq.stat(ch, :, :)) .^ 2) + (numel(subject_list) - 1));
        adj_petasq = petasq - (1 - petasq) .* (1 / (numel(subject_list) - 1));
        adjpetasq_seq_sti(ch, :, :) = adj_petasq;
    end
    
    % Average effect sizes across chans
    pd_tut_fix_adjpetasq = squeeze(mean(adjpetasq_tut_fix, 1));
    pd_tut_sti_adjpetasq = squeeze(mean(adjpetasq_tut_sti, 1));
    pd_seq_sti_adjpetasq = squeeze(mean(adjpetasq_seq_sti, 1));

    % Percentage of chans significant
    pd_tut_fix_percchanssig = squeeze(mean(clusts(1).idx, 1)) + squeeze(mean(clusts(2).idx, 1)) * 100;
    pd_tut_sti_percchanssig = squeeze(mean(clusts(3).idx, 1)) + squeeze(mean(clusts(4).idx, 1)) * 100;
    pd_seq_sti_percchanssig = squeeze(mean(clusts(5).idx, 1)) * 100;

    % Average in tf space per condition
    pd_tut_fix_tfpow_fix0 = squeeze(mean(GA_pow_fix0.powspctrm, [1, 2]));
    pd_tut_fix_tfpow_fix1 = squeeze(mean(GA_pow_fix1.powspctrm, [1, 2]));
    pd_tut_sti_tfpow_sti0 = squeeze(mean(GA_pow_sti0.powspctrm, [1, 2]));
    pd_tut_sti_tfpow_sti1 = squeeze(mean(GA_pow_sti1.powspctrm, [1, 2]));
    pd_seq_sti_tfpow_stay = squeeze(mean(GA_pow_stay.powspctrm, [1, 2]));
    pd_seq_sti_tfpow_swit = squeeze(mean(GA_pow_switch.powspctrm, [1, 2]));

    % Save time and freq vectors for veusz
    dlmwrite([PATH_PLOT, 'time_fix.csv'], stat_pow_fix.time);
    dlmwrite([PATH_PLOT, 'time_sti.csv'], stat_pow_sti_tut.time);
    dlmwrite([PATH_PLOT, 'freqs.csv'], stat_pow_fix.freq);
    
    % Save tf plot data for veusz
    dlmwrite([PATH_PLOT, 'pd_tut_fix_contour.csv'], pd_tut_fix_contour);
    dlmwrite([PATH_PLOT, 'pd_tut_sti_contour.csv'], pd_tut_sti_contour);
    dlmwrite([PATH_PLOT, 'pd_seq_sti_contour.csv'], pd_seq_sti_contour);
    dlmwrite([PATH_PLOT, 'pd_tut_fix_adjpetasq.csv'], pd_tut_fix_adjpetasq);
    dlmwrite([PATH_PLOT, 'pd_tut_sti_adjpetasq.csv'], pd_tut_sti_adjpetasq);
    dlmwrite([PATH_PLOT, 'pd_seq_sti_adjpetasq.csv'], pd_seq_sti_adjpetasq);
    dlmwrite([PATH_PLOT, 'pd_tut_fix_percchanssig.csv'], pd_tut_fix_percchanssig);
    dlmwrite([PATH_PLOT, 'pd_tut_sti_percchanssig.csv'], pd_tut_sti_percchanssig);
    dlmwrite([PATH_PLOT, 'pd_seq_sti_percchanssig.csv'], pd_seq_sti_percchanssig);
    dlmwrite([PATH_PLOT, 'pd_tut_fix_tfpow_fix0.csv'], pd_tut_fix_tfpow_fix0);
    dlmwrite([PATH_PLOT, 'pd_tut_fix_tfpow_fix1.csv'], pd_tut_fix_tfpow_fix1);
    dlmwrite([PATH_PLOT, 'pd_tut_sti_tfpow_sti0.csv'], pd_tut_sti_tfpow_sti0);
    dlmwrite([PATH_PLOT, 'pd_tut_sti_tfpow_sti1.csv'], pd_tut_sti_tfpow_sti1);
    dlmwrite([PATH_PLOT, 'pd_seq_sti_tfpow_stay.csv'], pd_seq_sti_tfpow_stay);
    dlmwrite([PATH_PLOT, 'pd_seq_sti_tfpow_swit.csv'], pd_seq_sti_tfpow_swit);

    % For topo: Proportion of tf points that have significant cluster contribution
    topo_percsig_clust_1 = sum(clusts(1).idx, [2, 3]) / sum(logical(squeeze(mean(clusts(1).idx, 1))), [1, 2]) * 100;
    topo_percsig_clust_2 = sum(clusts(2).idx, [2, 3]) / sum(logical(squeeze(mean(clusts(2).idx, 1))), [1, 2]) * 100;
    topo_percsig_clust_3 = sum(clusts(3).idx, [2, 3]) / sum(logical(squeeze(mean(clusts(3).idx, 1))), [1, 2]) * 100;
    topo_percsig_clust_4 = sum(clusts(4).idx, [2, 3]) / sum(logical(squeeze(mean(clusts(4).idx, 1))), [1, 2]) * 100;
    topo_percsig_clust_5 = sum(clusts(5).idx, [2, 3]) / sum(logical(squeeze(mean(clusts(4).idx, 1))), [1, 2]) * 100;

    % For topo: Average Effect size in cluster per channel
    for ch = 1 : 32
        topo_adjpetasq_clust_1(ch) = mean(adjpetasq_tut_fix(ch, logical(squeeze(mean(clusts(1).idx, 1)))));
        topo_adjpetasq_clust_2(ch) = mean(adjpetasq_tut_fix(ch, logical(squeeze(mean(clusts(2).idx, 1)))));
        topo_adjpetasq_clust_3(ch) = mean(adjpetasq_tut_sti(ch, logical(squeeze(mean(clusts(3).idx, 1)))));
        topo_adjpetasq_clust_4(ch) = mean(adjpetasq_tut_sti(ch, logical(squeeze(mean(clusts(4).idx, 1)))));
        topo_adjpetasq_clust_5(ch) = mean(adjpetasq_seq_sti(ch, logical(squeeze(mean(clusts(5).idx, 1)))));
    end

    % Foor topo: tf power differences per cluster
    tmp = squeeze(mean(GA_pow_fix1.powspctrm - GA_pow_fix0.powspctrm, 1));
    topo_powdiff_clust_1 = mean(tmp(:, logical(squeeze(mean(clusts(1).idx, 1)))), 2);
    topo_powdiff_clust_2 = mean(tmp(:, logical(squeeze(mean(clusts(2).idx, 1)))), 2);
    tmp = squeeze(mean(GA_pow_sti1.powspctrm - GA_pow_sti0.powspctrm, 1));
    topo_powdiff_clust_3 = mean(tmp(:, logical(squeeze(mean(clusts(3).idx, 1)))), 2);
    topo_powdiff_clust_4 = mean(tmp(:, logical(squeeze(mean(clusts(4).idx, 1)))), 2);
    tmp = squeeze(mean(GA_pow_switch.powspctrm - GA_pow_stay.powspctrm, 1));
    topo_powdiff_clust_5 = mean(tmp(:, logical(squeeze(mean(clusts(5).idx, 1)))), 2);

    % Plot average effect size for each cluster as topo
    markercolor = 'k';
    cmap = 'jet';
    clim = [-0.35, 0.35];
    figure('Visible', 'off'); clf;
    pd = topo_adjpetasq_clust_1;
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {clusts(1).chans_sig, 'p', markercolor, 14, 1});
    colormap(cmap);
    caxis(clim);
    saveas(gcf, [PATH_PLOT 'topo_adjpetasq_clust_1.png']);
    figure('Visible', 'off'); clf;
    pd = topo_adjpetasq_clust_2;
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {clusts(2).chans_sig, 'p', markercolor, 14, 1});
    colormap(cmap);
    caxis(clim);
    saveas(gcf, [PATH_PLOT 'topo_adjpetasq_clust_2.png']);
    figure('Visible', 'off'); clf;
    pd = topo_adjpetasq_clust_3;
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {clusts(3).chans_sig, 'p', markercolor, 14, 1});
    colormap(cmap);
    caxis(clim);
    saveas(gcf, [PATH_PLOT 'topo_adjpetasq_clust_3.png']);
    figure('Visible', 'off'); clf;
    pd = topo_adjpetasq_clust_4;
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {clusts(4).chans_sig, 'p', markercolor, 14, 1});
    colormap(cmap);
    caxis(clim);
    saveas(gcf, [PATH_PLOT 'topo_adjpetasq_clust_4.png']);
    pd = topo_adjpetasq_clust_5;
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {clusts(5).chans_sig, 'p', markercolor, 14, 1});
    colormap(cmap);
    caxis(clim);
    saveas(gcf, [PATH_PLOT 'topo_adjpetasq_clust_5.png']);

    % Cluster power differences for each channel as topo
    markercolor = 'k';
    cmap = 'jet';
    clim = [-1, 1];
    figure('Visible', 'off'); clf;
    pd = topo_powdiff_clust_1;
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {clusts(1).chans_sig, 'p', markercolor, 14, 1});
    colormap(cmap);
    caxis(clim);
    saveas(gcf, [PATH_PLOT 'topo_powdiff_clust_1.png']);
    figure('Visible', 'off'); clf;
    pd = topo_powdiff_clust_2;
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {clusts(2).chans_sig, 'p', markercolor, 14, 1});
    colormap(cmap);
    caxis(clim);
    saveas(gcf, [PATH_PLOT 'topo_powdiff_clust_2.png']);
    figure('Visible', 'off'); clf;
    pd = topo_powdiff_clust_3;
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {clusts(3).chans_sig, 'p', markercolor, 14, 1});
    colormap(cmap);
    caxis(clim);
    saveas(gcf, [PATH_PLOT 'topo_powdiff_clust_3.png']);
    figure('Visible', 'off'); clf;
    pd = topo_powdiff_clust_4;
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {clusts(4).chans_sig, 'p', markercolor, 14, 1});
    colormap(cmap);
    caxis(clim);
    saveas(gcf, [PATH_PLOT 'topo_powdiff_clust_4.png']);
    pd = topo_powdiff_clust_5;
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {clusts(5).chans_sig, 'p', markercolor, 14, 1});
    colormap(cmap);
    caxis(clim);
    saveas(gcf, [PATH_PLOT 'topo_powdiff_clust_5.png']);
   
end % End part7

% ======================= PART8: GATHERING SINGLE TRIAL INFO ===================================================================================================

% Preprocessing
if ismember('part8', to_execute)

    % Get chanlocs
    addpath(PATH_EEGLAB);
    eeglab;
    EEG = pop_loadset('filename', [subject_list{1} '_autocleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');
    chanlocs = EEG.chanlocs;

    % Load cluster struct
    load([PATH_CLUSTSTATS 'significant_clusters.mat']);

    % Load powcube lats
    load([PATH_TFDECOMP 'powcube_latencies']);
    pruned_segs = [-1000, 1500];
    time_idx_fix = dsearchn(tf_times', pruned_segs(1)) : dsearchn(tf_times', pruned_segs(2));
    pruned_segs = [-500, 1500];
    time_idx_sti = dsearchn(tf_times', pruned_segs(1)) : dsearchn(tf_times', pruned_segs(2));

    % Load data
    for s = 1 : length(subject_list)

        subject = subject_list{s};
        fprintf('\nLoad data subject %i/%i\n', s, numel(subject_list));

        % Load stuff
        POW = [];
        for ch = 1 : 32
            load([PATH_TFDECOMP 'powcube_' subject '_chan_' num2str(ch)]);
            POW(ch, :, :, :) = powcube;
        end
        load([PATH_TFDECOMP 'trialinfo_' subject]);

        % Identify corresponding events
        eids = unique(trialinfo(:, 7));

        % Iterate trials
        T = [];
        cnt = 0;
        for t = 1 : numel(eids)

            cnt = cnt + 1;

            % Get single trials for fix and sti segs
            trial_idx_fix = find(trialinfo(:, 7) == eids(t) & trialinfo(:, 2) == 0);
            trial_idx_sti = find(trialinfo(:, 7) == eids(t) & trialinfo(:, 2) == 1);
            trial_fix = squeeze(POW(:, :, time_idx_fix, trial_idx_fix));
            trial_sti = squeeze(POW(:, :, time_idx_sti, trial_idx_sti));

            % A matrix with all subject trilas                                               % Columns:
            T(cnt, :) = [str2num(subject(end - 3 : end)),...                                 %  1 = id
                         trialinfo(trialinfo(:, 7) == eids(t) & trialinfo(:, 2) == 0, 3),... %  2 = tut
                         trialinfo(trialinfo(:, 7) == eids(t) & trialinfo(:, 2) == 0, 4),... %  3 = switch
                         trialinfo(trialinfo(:, 7) == eids(t) & trialinfo(:, 2) == 0, 5),... %  4 = rt
                         trialinfo(trialinfo(:, 7) == eids(t) & trialinfo(:, 2) == 0, 6),... %  5 = accuracy
                         trialinfo(trialinfo(:, 7) == eids(t) & trialinfo(:, 2) == 0, 8),... %  6 = position
                         mean(trial_fix(clusts(1).idx)),...                                  %  7 = Cluster 1 average power
                         mean(trial_fix(clusts(2).idx)),...                                  %  8 = Cluster 2 average power
                         mean(trial_sti(clusts(3).idx)),...                                  %  9 = Cluster 3 average power
                         mean(trial_sti(clusts(4).idx)),...                                  %  10 = Cluster 4 average power
                         mean(trial_sti(clusts(5).idx)),...                                  %  10 = Cluster 5 average power
                        ];
        end

        % Merge things...
        if s == 1
            M = T;
        else
            M = [M; T];
        end

    end % End subit

    % Save single trial data
    dlmwrite([PATH_CLUSTSTATS 'single_trial_data.csv'], M);

end % End part8

% ======================= PART9: BEHAVIORAL ANALYSIS ==================================================================================

% Preprocessing
if ismember('part9', to_execute)

    % Load single trial data
    data = dlmread([PATH_CLUSTSTATS 'single_trial_data.csv']);

    % Remove outliers
    crit = nanmedian(data(:, 4)) + nanstd(data(:, 4)) * 3;
    data(data(:, 5) ~= 0 & (data(:, 4) < 200 | data(:, 4) > crit), :) = [];

    % For rt analysis only correct trials
    data_correct = data(data(:, 5) ~= 0, :);

    % Average tut and switch conditions for plotting
    res = [];
    for tut = 0 : 1
        for seq = 0 : 1
            rt = mean(data_correct(data_correct(:, 2) == tut & data_correct(:, 3) == seq, 4));
            ac = mean(data(data(:, 2) == tut & data(:, 3) == seq, 5)) * 100;
            X = [];
            Y = [];
            for s = 1 : length(subject_list)
                subject = subject_list{s};
                id = str2num(subject(end - 3 : end));
                X(s) = mean(data_correct(data_correct(:, 2) == tut & data_correct(:, 3) == seq & data_correct(:, 1) == id, 4)); 
                Y(s) = mean(data(data(:, 2) == tut & data(:, 3) == seq & data(:, 1) == id, 5)) * 100; 
            end
            rt_std = std(X);
            ac_std = std(Y);
            res(seq + 1, (tut * 4 + 1) : (tut * 4 + 4)) = [rt, rt_std, ac, ac_std];
        end
    end
    dlmwrite([PATH_PLOT 'behavior.csv'], res, 'delimiter', '\t');
    dlmwrite([PATH_PLOT 'xax.csv'], [1, 2]);

    % Descriptives for seq only
    res_seq = [];
    for seq = 0 : 1
        rt = mean(data_correct(data_correct(:, 3) == seq, 4));
        ac = mean(data(data(:, 3) == seq, 5)) * 100;
        X = [];
        Y = [];
        for s = 1 : length(subject_list)
            subject = subject_list{s};
            id = str2num(subject(end - 3 : end));
            X(s) = mean(data_correct(data_correct(:, 3) == seq & data_correct(:, 1) == id, 4)); 
            Y(s) = mean(data(data(:, 3) == seq & data(:, 1) == id, 5)) * 100; 
        end
        rt_std = std(X);
        ac_std = std(Y);
        res_seq(seq + 1, :) = [rt, rt_std, ac, ac_std];
    end

    % Descriptives for tut only
    res_tut = [];
    for tut = 0 : 1
        rt = mean(data_correct(data_correct(:, 2) == tut, 4));
        ac = mean(data(data(:, 2) == tut, 5)) * 100;
        X = [];
        Y = [];
        for s = 1 : length(subject_list)
            subject = subject_list{s};
            id = str2num(subject(end - 3 : end));
            X(s) = mean(data_correct(data_correct(:, 2) == tut & data_correct(:, 1) == id, 4)); 
            Y(s) = mean(data(data(:, 2) == tut & data(:, 1) == id, 5)) * 100; 
        end
        rt_std = std(X);
        ac_std = std(Y);
        res_tut(tut + 1, :) = [rt, rt_std, ac, ac_std];
    end

    % LME response times
    M = data_correct;
	varnames = {'id', 'tut' , 'swi', 'rt', 'acc'};
	tbl = table(M(:, 1), M(:, 2), M(:, 3), M(:, 4), M(:, 5), 'VariableNames', varnames);
	tbl.id = nominal(tbl.id);
    tbl.tut = nominal(tbl.tut, {'tut0', 'tut1'});
    tbl.swi = nominal(tbl.swi, {'rep', 'swi'});
    lme_rt = fitlme(tbl, 'rt ~ tut*swi + (1|id)');

    % LME accuracy
    M = data;
    varnames = {'id', 'tut' , 'swi', 'rt', 'acc'};
    tbl = table(M(:, 1), M(:, 2), M(:, 3), M(:, 4), M(:, 5), 'VariableNames', varnames);
    tbl.id = nominal(tbl.id);
    tbl.tut = nominal(tbl.tut, {'tut0', 'tut1'});
    tbl.swi = nominal(tbl.swi, {'rep', 'swi'});
    lme_acc = fitlme(tbl, 'acc ~ tut*swi + (1|id)');

end % End part9