%% digital predistortion linearization of PA in WebLab
% 
% 20/11/2019

clear all;close all;clc;

%% Frequency Object
Window='Blackman-Harris';%'blackman';
SegLength=2^10;
overlap=25;
Hs = spectrum.welch(Window,SegLength,overlap);

FuncMode='Testbench'; %DataAcquis Testbench ILC
%% Display control
EnAMbL='on';
EnSpec='on';
% EnTime='off';

%% GetStimulus
load('PAinLTE20MHz') % load your own baseband data

%% Get PA model

PAPRin=papr(PAin);
RMSin=-8.5-PAPRin-2;
[PAout, RMSout, Idc, Vdc] = RFWebLab_PA_meas_v1_2(PAin, RMSin); 
PAout=timealign(PAin,PAout);
        DisplayOptions.InterCorrArg.Enable='off';
        DisplayOptions.IQ.Scope=1e5;
        DisplayOptions.IQ.Enable='off';
        DisplayOptions.NMSE.Enable='on';
    Data.In=PAin;
    Data.Out=PAout;
    Data.ALimLinIn=0.2;
    
    PA=amampm(Data,EnAMbL);
%% Predistortion identification: Applying IBO

Backoff=20*log10(PA.LimitPD/max(abs(PAin)));

% Set your input average power
RMSin=RMSin+Backoff;

[PAout, RMSout, Idc, Vdc] = RFWebLab_PA_meas_v1_2(PAin, RMSin);

PAout=timealign(PAin,PAout);
    %% Input/Output Pwr spectral density 
[ACPRin, PSDin]=acpr(PAin,Fs,ACPR);
[ACPRout, PSDout]=acpr(PAout,Fs,ACPR);
X=PSDin.Data;
PXin=10*log10(X);
Y=PSDout.Data;
PYout=10*log10(Y);
f=PSDout.Frequencies;

fprintf(['\n\t The output power before linearization is',...
    ' equal to %.2f dBm\n'],...
    RMSout)


%% Model PD first stage
clear ModelPD

PD.In=PAin;
PD.BW=BW;
%% Iterations
section=10e3; % number of samples for each iteration in indirect learning
N=length(PAin)-rem(length(PAin),section);

clear Data
Data.Fs=Fs;
Data.ACPR=ACPR;

Algorithm.NbIterPerStage=4;
Algorithm.DampingNewtonFactor=0.7;
Algorithm.NbSampPerIter=section;

Stimulus.wf=PAin;
Stimulus.RMSin=RMSin;
Stimulus.Fs=Fs;
Stimulus.ACPR=ACPR;

%% Launch DPD Test
tic;

SystIter=DPD(Stimulus,ACPRout,Algorithm,PD); % change in file "DPD.m" to implement your DPD algorithm

timerVal=toc;
fprintf(['\n\t Execution time: ', num2str(timerVal), ' seconds'])

DisplayFigures;
