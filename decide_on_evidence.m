function [decisions] = decide_on_evidence(ipp, trial_labels, threshold, trespassing_eval, classes_labels)
%   [decisions] = decide_on_evidence(ipp, trial_labels, threshold, trespassing_eval, classes_labels)
%	Evaluates the output of an evidence accumulation paradigm applied to a
%	flow of posterior probabilities (of a binary classificator)
%
%   Input arguments:
%       -ipp: [samples x 1] matrix of integrated posterior probabilies,
%        values in [0,1]
%        First class:
%       -trial_labels: [samples x 1] labels vector that divide the
%        integrated posterior probabilies into trials (11...1122...22...)
%       -threshold: value of threshold for which the trial is assigned to a
%        class (second class has thresold = 1-threshold)
%       -trespassing_eval: (possible values: 'first', 'last')
%        Sets how to assign each trial to a class.
%        If 'first' fuction assigns each trial to the class which threshold
%        is crossed first, ('last' is analogous)
%       -classes_labels: [first_class_label, second_class_label, undecided_class]'
%
%   Output arguments:
%       -decisions [trials x 1] contains the decision associated to each trial
%
    undecided_class = classes_labels(1,3);
    class_1 = classes_labels(1,1);
    class_2 = classes_labels(1,2);
    num_trials = max(trial_labels)-min(trial_labels)+1;
    decisions = nan(num_trials, 1);
    for trId = 1:num_trials
        curr_ipp = ipp(trial_labels==trId);
        
        endpoint = find( (curr_ipp >= threshold) | (curr_ipp <= 1 - threshold), 1, trespassing_eval );
    
        if(isempty(endpoint))
            decisions(trId) = undecided_class;
            continue;
        end
    
        if(curr_ipp(endpoint) >= threshold)
            decisions(trId) = class_1;
        elseif (curr_ipp(endpoint) <= threshold)
            decisions(trId) = class_2;
        end
    end

end

