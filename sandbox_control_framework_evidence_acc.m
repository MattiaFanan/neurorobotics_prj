% Control Framework test / sandbox

clearvars;clc;

test_offline_file = 'ah7.20170613.162934.offline.mi.mi_bhbf.gdf.mat';
%mnemonic codes
code.fixation = 786;
code.cue_BH = 773;
code.cue_BF = 771;
code.cue_rest = 783;
code.feedback = 781;
mode.offline = 0;
mode.online = 1;
num_classes = 2;
data_path = './data/';
%sub-band of frequencies of psd we are interested in
selected_frequences = (4:2:48)';
num_features = 5;
features_filter = load('features_filter.mat').features_filter';

%Load OFFLINE file
load(strcat(data_path, test_offline_file));
%the file shoud contain a struct named "data" containing a signal and its
%header (data.s and data.h)

%% EEG to PSD

signal = data.s;
header = data.h;

psd_signal = psd_extraction(signal, header);

%% log and sub-frequences extraction

psd_normalized = log(psd_signal.PSD);
freqs = nan;
%psd is windows x freq x channels
original_frequences = psd_signal.frequences;
%extract selected subfrequences
[psd_signal.PSD, frequences] = extractFrequences(psd_normalized, original_frequences, selected_frequences);

%% convert PSDs into dataset for the classifiers
PSD_DATA{1} = psd_signal;
PSD_DATA{1}.modality = mode.online;
[X, cue_type_labels, trial_labels, modality_labels, selected_freq_chan_index , fisher_score_run] = psd2features(PSD_DATA, num_features, features_filter);

%% compute mean fisher's score
mean_fisher_score = mean(fisher_score_run,3);

%% generate map of selected features
features_mask = zeros(size(mean_fisher_score));

for k = 1 : length(selected_freq_chan_index)
    feature_index = selected_freq_chan_index{k};
    features_mask(feature_index(1), feature_index(2)) = 1;
end

%% Train a classifier
train_set = X(cue_type_labels == code.cue_BH |cue_type_labels == code.cue_BF);
label_set = cue_type_labels(cue_type_labels == code.cue_BH | cue_type_labels == code.cue_BF);

classifier = train_binary_model(train_set, label_set); %probably SVM

%% ONLINE BITCHEEEEEESSSS
%load online file
load(strcat(data_path, 'ah7.20170613.170929.online.mi.mi_bhbf.ema.gdf.mat'));
% now data contains the online signal

%% EEG to PSD

signal = data.s;
header = data.h;

psd_signal = psd_extraction(signal, header);

%% log and sub-frequences extraction

psd_normalized = log(psd_signal.PSD);
freqs = nan;
%psd is windows x freq x channels
original_frequences = psd_signal.frequences;
%extract selected subfrequences
[psd_signal.PSD, frequences] = extractFrequences(psd_normalized, original_frequences, selected_frequences);

%% Labeling (online) data
num_windows = size(psd_signal.PSD, 1);
[cue_type_labels, trial_labels] = labelData(psd_signal.EVENT, num_windows);

trials = zeros(size(cue_type_labels,1), 1);
k=1;
for i = 2:length(cue_type_labels)
    if (cue_type_labels(i,1) ~= 0)
        trials(i,1) = k;
    end
    
    if (cue_type_labels(i,1) == 0 && cue_type_labels(i-1,1) ~= 0)
        k = k+1;
    end
        
end
cue_type_labels = cue_type_labels(cue_type_labels ~=0);
trials = trials(trials ~= 0);
% 11...11 22...22 ...
num_trials = max(trials);

%% Features extraction
X_test_orig = extractFeatures(psd_signal.PSD, selected_freq_chan_index); %all
%don't consider resting periods for Single sample acc
X_test = X_test_orig(cue_type_labels == code.cue_BH | cue_type_labels == code.cue_BF);
% [windows x features]

%% Classification of online data
[computed_labels, post_probabilities, ~] = predict(classifier, X_test);

single_sample_acc = 100*sum(computed_labels == cue_type_labels(cue_type_labels == code.cue_BH | cue_type_labels == code.cue_BF))./length(computed_labels);

disp('Single Sample accuracy: ')
disp(single_sample_acc)

%reconsider resting periods for evidence accumulation
X_test = X_test_orig(cue_type_labels == code.cue_BH | cue_type_labels == code.cue_BF | cue_type_labels==code.cue_rest);
[computed_labels, post_probabilities, ~] = predict(classifier, X_test);

%It's a double job, but just for the sake of testing

%% Evidence accumulation
%Applying exponential smoothing

disp('[proc] + Evidence accumulation (exponential smoothing)');

num_samples = size(post_probabilities, 1);

ipp = 0.5*ones(size(post_probabilities, 1), 1);
alpha = 0.97;
sample_offset = 0;


for trial_n = 1:num_trials
    curr_trial_post_prob = post_probabilities(trials == trial_n, :);
    curr_trial_computed_labels = computed_labels(trials == trial_n);
    for j = 2:size(curr_trial_post_prob)
        ipp(sample_offset+j,1) = ipp(sample_offset+j-1,1).*alpha + curr_trial_post_prob(j, 1).*(1-alpha);
    end
    sample_offset = sample_offset + j;
    
end

%% Plot accumulated evidence and raw probabilities
fig1 = figure;

CueClasses    = [code.cue_BF code.cue_rest code.cue_BH];
LbClasses     = {'both feet', 'rest', 'both hands'};
ValueClasses  = [1 0.5 0];
Threshold     = 0.7;

SelTrial = 1;

GreyColor = [150 150 150]/255;
LineColors = {'b', 'g', 'r'};

hold on;
trial_indices = trials==SelTrial;
% Plotting raw probabilities
plot(post_probabilities(trial_indices, 1), 'o', 'Color', GreyColor);

% Plotting accumulutated evidence
plot(ipp(trial_indices), 'k', 'LineWidth', 2);

% Plotting actual target class
class = CueClasses == (cue_type_labels(find(trials == SelTrial, 1, 'first')));
yline(ValueClasses(class), LineColors{class}, 'LineWidth', 5);

% Plotting 0.5 line
yline(0.5, '--k');

% Plotting thresholds
yline(Threshold, 'k', 'Th_{1}');
yline(1-Threshold, 'k', 'Th_{2}');
hold off;

grid on;
ylim([0 1]);
xlim([1 sum(trial_indices)]);
legend('raw prob', 'integrated prob');
ylabel('probability/control')
xlabel('sample');
title(['Trial ' num2str(SelTrial) '/' num2str(num_trials) ' - Class ' LbClasses{class} ' (' num2str(CueClasses(class)) ')']);


%% Compute performances
ActualClass = psd_signal.EVENT.TYP(psd_signal.EVENT.TYP == code.cue_BF | psd_signal.EVENT.TYP == code.cue_BH | psd_signal.EVENT.TYP == code.cue_rest);
Decision = nan(num_trials, 1);

for trId = 1:num_trials
    curr_ipp = ipp(trials==trId);
    
    endpoint = find( (curr_ipp >= Threshold) | (curr_ipp <= 1 - Threshold), 1, 'last' );
    
    if(isempty(endpoint))
        Decision(trId) = 783;
        continue;
    end
    
    if(curr_ipp(endpoint) >= Threshold)
        Decision(trId) = 771;
    elseif (curr_ipp(endpoint) <= Threshold)
        Decision(trId) = 773;
    end
end

% Removing Rest trials
ActiveTrials = (ActualClass ~= code.cue_rest);
RestTrials = (ActualClass == code.cue_rest);

PerfActive  = 100 * (sum(ActualClass(ActiveTrials) == Decision(ActiveTrials))./sum(ActiveTrials))
PerfResting = 100 * (sum(ActualClass(RestTrials) == Decision(RestTrials))./sum(RestTrials))

RejTrials = Decision == 783;

PerfActive_rej = 100 * (sum(ActualClass(ActiveTrials & ~RejTrials) == Decision(ActiveTrials & ~RejTrials))./sum(ActiveTrials & ~RejTrials))


