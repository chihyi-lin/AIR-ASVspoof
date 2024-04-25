clear; close all; clc;

% This code is modified from the baseline system for ASVspoof 2019

% add required libraries to the path
addpath(genpath('LFCC'));
addpath(genpath('CQCC_v1.0'));

% set here the experiment to run (access and feature type)
access_type = 'LA';
feature_type = 'LFCC';

% set paths to the wave files and protocols
pathToFeatures = horzcat('/Users/chihyi/Documents/CL_WS_24/Speech_tech/project/AIR-ASVspoof/in_the_wild_MidFeatures/');

pathToDatabase = horzcat('/Users/chihyi/Documents/CL_WS_24/Speech_tech/project/AIR-ASVspoof/release_in_the_wild');
evalProtocolFile = fullfile(pathToDatabase, 'meta.csv');

% read eval protocol
evalfileID = fopen(evalProtocolFile);
evalprotocol = textscan(evalfileID, '%s%s%s', 'Delimiter', ',');
fclose(evalfileID);
evalfilelist = cellfun(@(x) regexp(x, '\d+', 'match', 'once'),evalprotocol{1}, 'UniformOutput', false);


%% Feature extraction for evaluation data

% extract features for training data and store them
disp('Extracting features for evaluation data...');
for i=2:length(evalfilelist)
    filePath = fullfile(pathToDatabase, [evalfilelist{i} '.wav']);
    [x,fs] = audioread(filePath);
    [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
    LFCC = [stat delta double_delta]';
    filename_LFCC = fullfile(pathToFeatures, horzcat(evalfilelist{i}, '.mat'));
    parsave(filename_LFCC, LFCC)
    LFCC = [];
end
disp('Done!');


%% supplementary function
function parsave(fname, x)
    save(fname, 'x')
end
