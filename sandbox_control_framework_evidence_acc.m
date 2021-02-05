% Control Framework test / sandbox

clearvars;clc;

test_offline_file = 'ai6.20180316.153104.offline.mi.mi_bhbf.gdf';
%mnemonic codes
code.fixation = 786;
code.cue_BH = 773;
code.cue_BF = 771;
code.cue_rest = 783;
code.feedback = 781;
mode.offline = 0;
mode.online = 1;
num_classes = 2;
data_path = './data/ai6_micontinuous/';
%sub-band of frequencies of psd we are interested in
selected_frequences = (4:2:48)';
num_features = 5;
features_filter = load('features_filter.mat').features_filter';

%Load OFFLINE file
%load(strcat(data_path, test_offline_file));
%the file shoud contain a struct named "data" containing a signal and its
%header (data.s and data.h)

%% EEG to PSD

%signal = data.s;
%header = data.h;
[signal, header] = sload(strcat(data_path, test_offline_file));

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
train_set = X(cue_type_labels == code.cue_BH |cue_type_labels == code.cue_BF, :);
label_set = cue_type_labels(cue_type_labels == code.cue_BH | cue_type_labels == code.cue_BF);

classifier = train_binary_model(train_set, label_set); %probably SVM

%% ONLINE BITCHEEEEEESSSS
%load online file
[signal, header] = sload(strcat(data_path, 'ai6.20180316.161026.online.mi.mi_bhbf.dynamic.gdf'));
% now data contains the online signal

%% EEG to PSD
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
trial_labels = trial_labels(trial_labels  ~= 0); % 11...11 22...22 ...
cue_type_labels = cue_type_labels(cue_type_labels ~=0);
num_trials = max(trial_labels);

%% Features extraction
X_test_orig = extractFeatures(psd_signal.PSD, selected_freq_chan_index); %all
%don't consider resting periods for Single sample acc
X_test = X_test_orig(cue_type_labels == code.cue_BH | cue_type_labels == code.cue_BF, :);
% [windows x features]

%% Classification of online data
[computed_labels, ~, ~] = predict(classifier, X_test);

single_sample_acc = 100*sum(computed_labels == cue_type_labels(cue_type_labels == code.cue_BH | cue_type_labels == code.cue_BF))./length(computed_labels);

disp('Single Sample accuracy: ')
disp(single_sample_acc)

%reconsider resting periods for evidence accumulation
X_test = X_test_orig(cue_type_labels == code.cue_BH | cue_type_labels == code.cue_BF | cue_type_labels==code.cue_rest, :);
[computed_labels, post_probabilities, ~] = predict(classifier, X_test);

%It's a double job, but just for the sake of testing

%% Evidence accumulation
%Applying exponential smoothing

disp('[proc] + Evidence accumulation');
initial_value = 0.5;
alpha = 0.9;
beta = 0.7;
ipp = exponential_smoothing(post_probabilities, trial_labels, initial_value, alpha);
%ipp = dynamic_smoothing(post_probabilities, trial_labels, alpha, beta);


%% Plot accumulated evidence and raw probabilities
fig1 = figure;

CueClasses    = [code.cue_BF code.cue_rest code.cue_BH];
LbClasses     = {'both feet', 'rest', 'both hands'};
ValueClasses  = [1 0.5 0];
Threshold     = 0.7;

SelTrial = 10;

GreyColor = [150 150 150]/255;
LineColors = {'b', 'g', 'r'};

hold on;
trial_indices = trial_labels==SelTrial;
% Plotting raw probabilities
plot(post_probabilities(trial_indices, 1), 'o', 'Color', GreyColor);

% Plotting accumulutated evidence
plot(ipp(trial_indices), 'k', 'LineWidth', 2);

% Plotting actual target class
class = CueClasses == (cue_type_labels(find(trial_labels == SelTrial, 1, 'first')));
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
Decision = decide_on_evidence(ipp, trial_labels, Threshold, 'last', [code.cue_BF, code.cue_BH, code.cue_rest]);

% Removing Rest trials
ActiveTrials = (ActualClass ~= code.cue_rest);
RestTrials = (ActualClass == code.cue_rest);

PerfActive  = 100 * (sum(ActualClass(ActiveTrials) == Decision(ActiveTrials))./sum(ActiveTrials))
PerfResting = 100 * (sum(ActualClass(RestTrials) == Decision(RestTrials))./sum(RestTrials))

RejTrials = Decision == 783;

PerfActive_rej = 100 * (sum(ActualClass(ActiveTrials & ~RejTrials) == Decision(ActiveTrials & ~RejTrials))./sum(ActiveTrials & ~RejTrials))


