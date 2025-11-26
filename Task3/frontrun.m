%% Common code for all functions

restoredefaultpath;
p=pathup(LevelUp);
% disp(genpath(p))
addpath(genpath([p,'\Lib']))
clear all;close all;clc;
%% Old code
% path(path,'..\..\Lib\Stim');
% % PA model file
% path(path,'..\Lib\PAmodel');
% % Measurement function
% path(path,'..\Lib\Measure');
% % Misc
% path(path,'..\Lib\Misc');
% % LS algo
% path(path,'..\Lib\LS');
% % Results
% path(path,'..\lib\Result');
% % Experimental function
% path(path,'..\Trial');

%% Frequency Object
Window='Blackman-Harris';%'blackman';
SegLength=2^10;
overlap=25;
Hs = spectrum.welch(Window,SegLength,overlap);
