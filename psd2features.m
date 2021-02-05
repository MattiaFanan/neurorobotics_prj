function [data_set, selected_freq_chan_index , fisher_score_run] = psd2features(PSD, run_labels, cue_type_labels, classes, num_features ,feature_weights)
% [data_set, selected_freq_chan_index , fisher_score_run] = psd2features(PSD, run_labels, cue_type_labels, classes, num_features ,feature_weights)
%
% The function extracts the data to be used to train the classifiers (remember to normalize the PSD e.g. with log())
%
% Input arguments:
%   - PSD_data                  [windows x frequences x channels] matrix
%   containing PSD data
%   - run_labels                [windows x 1] vector that associates a run
%   number to each windows 
%   - cue_type_labels           [windows x 1] vector that associates a cue
%   task to each windows
%   -classes                    vector containing one code for each class
%   - num_features              number of features to extract
%   - feature_weights           (optional) weight to modify the priority of
%   the features extracted with respect to their fisher's score
%
% Output arguments:
%   - data_set                  [windows x feature] matrix that is the
%   dataset to use in training
%   - selected_freq_chan_index  cell array containing the coordinates of all 
%   selected features     
%   - fisher_score_run          [freqences x channels x runs] atrix
%   containing one fisher's score matrix for each run

%% Computing fisher score (for each run)

run_codes = unique(run_labels);
fisher_score_run = fisherScore(PSD, run_labels, cue_type_labels, run_codes, classes);
mean_fisher_score = mean(fisher_score_run, 3);

%% Features selection
if ~exist('feature_weights','var')
    %default weights are a matrix of all ones
    feature_weights = ones(size(mean_fisher_score));
end
filtered_mean_fisher = mean_fisher_score .* feature_weights;
selected_freq_chan_index = bestFeatures(filtered_mean_fisher, num_features);

%% Features extraction
data_set = extractFeatures(PSD , selected_freq_chan_index);
    
end

