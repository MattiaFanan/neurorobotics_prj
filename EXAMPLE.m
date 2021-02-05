clearvars;clc;

files{1} = 'ah7.20170613.161402.offline.mi.mi_bhbf.gdf';
files{2} = 'ah7.20170613.162331.offline.mi.mi_bhbf.gdf';
files{3} = 'ah7.20170613.162934.offline.mi.mi_bhbf.gdf';
%mnemonic codes
code.fixation = 786;
code.cue_BH = 773;
code.cue_BF = 771;
code.cue_rest = 783;
code.feedback = 781;
mode.offline = 0;
mode.online = 1;
data_path = './data/';
%sub-band of frequencies of psd we are interested in
selected_frequences = (4:2:48)';
num_features = 5;
features_filter = load('features_filter.mat').features_filter';

%% load EEG
disp('[io] + loading EEG files');

EEG_data = cell(size(files));

for file = 1 : length(files)
    [s,h] = sload(fullfile(data_path, files{file}));
    EEG_data{file}.signal = s;
    EEG_data{file}.header = h;
end

%% convert EEG into PSD

PSD_data = cell(size(EEG_data));

for eeg_i = 1 : length(EEG_data)
    signal = EEG_data{eeg_i}.signal;
    header = EEG_data{eeg_i}.header;
    PSD_data{eeg_i} = psd_extraction(signal, header);
    %adding modality info
    modality = nan ;
    if contains(files{eeg_i},"offline")
        modality = 0;
    elseif contains(files{eeg_i},"online")
        modality = 1;
    else
        error(['Unknown modality for file: ' eeg_i]);
    end
    PSD_data{eeg_i}.modality = modality;
end

%% log and sub-frequences extraction
freqs = nan;
%psd is windows x freq x channels
for psd_i = 1 : length(PSD_data)
    PSD = PSD_data{psd_i}.PSD;
    %apply log to normalize the distribution
    PSD = log(PSD);
    original_frequences = PSD_data{psd_i}.frequences;
    %extract selected subfrequences
    [PSD_data{psd_i}.PSD, PSD_data{psd_i}.frequences] = extractFrequences(PSD, original_frequences, selected_frequences);
    freqs = PSD_data{psd_i}.frequences;
end
%% convert PSDs into dataset for the classifiers

[X, cue_type_labels, trial_labels, modality_labels, selected_freq_chan_index , fisher_score_run] = psd2features(PSD_data, num_features, features_filter);

%% compute mean fisher's score

mean_fisher_score = mean(fisher_score_run,3);

%% filter mean fisher's score

filtered_mean_Fisher = mean_fisher_score .* features_filter;

%% generate map of selected features

features_mask = zeros(size(mean_fisher_score));

for k = 1 : length(selected_freq_chan_index)
    feature_index = selected_freq_chan_index{k};
    features_mask(feature_index(1), feature_index(2)) = 1;
end
%% Visualization Fisher score

channelLb  = {'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4'};
channelId  = 1:length(channelLb);

NumRuns = length(EEG_data);
NumFreqs = length(freqs);
NumChans = length(channelLb);
classId    = [773 771];
classLb    = {'both hands', 'both feet'};
nclasses   = length(classId);
modalityId = [0 1];
modalityLb = {'offline', 'online'};

disp('[proc] |- Visualizing fisher score');
OfflineRuns = [1 2 3];
climits = [];
handles = nan(NumRuns, 1);
fig1 = figure;

for rId = 1:length(OfflineRuns)
    subplot(2, 3, rId);
    imagesc(fisher_score_run(:, :, OfflineRuns(rId))');
    axis square;
    set(gca, 'XTick', 1:NumFreqs);
    set(gca, 'XTickLabel', freqs);
    set(gca, 'YTick', 1:NumChans);
    set(gca, 'YTickLabel', channelLb);
    xtickangle(-90);
    
    title(['Calibration run ' num2str(OfflineRuns(rId))]);
    
    climits = cat(2, climits, get(gca, 'CLim'));
    handles(OfflineRuns(rId)) = gca;
end

subplot(2, 3, rId+1);
imagesc(mean_fisher_score');
axis square;
set(gca, 'XTick', 1:NumFreqs);
set(gca, 'XTickLabel', freqs);
set(gca, 'YTick', 1:NumChans);
set(gca, 'YTickLabel', channelLb);
xtickangle(-90);

title(['mean over #runs = ' num2str(OfflineRuns(rId))]);

climits = cat(2, climits, get(gca, 'CLim'));
handles(OfflineRuns(rId)) = gca;

subplot(2, 3, rId+2);
imagesc(filtered_mean_Fisher');
axis square;
set(gca, 'XTick', 1:NumFreqs);
set(gca, 'XTickLabel', freqs);
set(gca, 'YTick', 1:NumChans);
set(gca, 'YTickLabel', channelLb);
xtickangle(-90);

title(['filtered mean over #runs = ' num2str(OfflineRuns(rId))]);

climits = cat(2, climits, get(gca, 'CLim'));
handles(OfflineRuns(rId)) = gca;


subplot(2, 3, rId+3);
imagesc(features_mask');
axis square;
set(gca, 'XTick', 1:NumFreqs);
set(gca, 'XTickLabel', freqs);
set(gca, 'YTick', 1:NumChans);
set(gca, 'YTickLabel', channelLb);
xtickangle(-90);

title('features mask');


set(handles, 'CLim', [min(min(climits)) max(max(climits))]);

sgtitle('Fisher score');

train_set = X(cue_type_labels == code.cue_BH |cue_type_labels == code.cue_BF);
label_set = cue_type_labels(cue_type_labels == code.cue_BH | cue_type_labels == code.cue_BF);

probably_SVM = train_binary_model(train_set, label_set);

