%% GMP Model Explorer
%
% In this program we will explore GMP model dpd parametrization on
% the Doherty PA model using 20MHz LTE signal.
% 12/06/2012
% updated 27/09/2015 by WANG Siqi

if ~exist('NightExecution','var')
    clear all;close all;clc;
    LevelUp=2;
    frontrun;
end
%% Display control
DisplayOption.Enable='off';
DisplayOption.NMSE='on';
DisplayOption.Spectra='on';
DisplayOption.AMAM='on';
DisplayOption.ACPR='on';
DisplayOption.EVM='on';

%% Get PA model
PAmodel=1;
PAGainType='peak'; %peak linear

if PAmodel==1
    load Matlab/A2I20D040N_Synchronizer_ARELIS_LTE20MHz_2GHz %Synchronizer5dBm Synchronizer1dBm
    %     Synchronizer2dBm_LTE5MHz Synchronizer3dBm_LTE10MHz
    %     Synchronizer_OFDM_11dB Synchronizer_M1dBm
    %     Synchronizer_10MHzclipped_5dBm Synchronizer_5MHzclipped_5dBm
elseif PAmodel==2
    load Matlab/AmpliDoherty_666MHz_BW8
elseif PAmodel==3
    load Matlab/X5port_noclip_nofilt_6_9MHz_52_9dBm
else
    load Matlab/X8port_noclip_nofilt_6_9MHz_51_7dBm
end
%AmpliDoherty_666MHz_BW8 Synchronizer5dB X5port_noclip_nofilt_6_9MHz_52_9dBm X8port_noclip_nofilt_6_9MHz_51_7dBm
% load Matlab\DohertyPAModel\DohertyPAGMP2_2_NMSE
% load PAinLTE20MHz
% x=resample(PAin,200e6,Fs);
% Synchronizer.Input.Fs=200e6;
% Synchronizer.Input.BW=BW;
% x=fixpwr(x,5); % -2 dBm
% y=gmp(x,DohertyPAMP);
% Synchronizer.Input.IQdata=x;
% Synchronizer.InputRef.IQdata=x;
% Synchronizer.Output.IQdata=fixpwr(y,pwrdbm(y)-30);

%% Input/Output Pwr spectral density
% NbPts=100e3;
% x=PAin(1:NbPts);
% y=PAout(1:NbPts);

Fs=Synchronizer.Input.Fs;
ACPR.BW=Synchronizer.Input.BW;
ACPR.Offset=1.1*Synchronizer.Input.BW;
x=Synchronizer.InputRef.IQdata;
x=fixpwr(x,5); % -2 dBm
PAin=x;
u=x;
y=Synchronizer.Output.IQdata;
y=fixpwr(y,pwrdbm(y)+30); % 47 dBm
PAout=y;
PSDx=psd(Hs,x,'Fs',Fs,'CenterDC',true);
X=PSDx.Data;
PX = 10*log10(X/max(X));
PSDy=psd(Hs,y,'Fs',Fs,'CenterDC',true);
Y=PSDy.Data;
PY = 10*log10(Y/max(Y));
f=PSDx.Frequencies;
if strcmp(PAGainType,'linear')
    Iv=abs(x)>0.3*max(abs(x)) & abs(x)<max(abs(x));
    PA.G=mean(abs(y(Iv))./abs(x(Iv)));
    
    [PA.InSatV,I]=max(abs(x));
    PA.OutSatV=abs(y(I));
    [~,I]=min(abs(abs(x*PA.G)-PA.OutSatV));
    PA.InLPASatV=abs(x(I));
    Inew=x<PA.InLPASatV;
    x=x(Inew);
    y=y(Inew);
else
    PA.G=max(abs(y))./max(abs(x));
end

%% ACPR parameters
[ACPRin, PSDin]=acpr(PAin,Fs,ACPR,Hs);
[ACPRout, PSDout]=acpr(PAout,Fs,ACPR,Hs);
X=PSDin.Data;
PXin=10*log10(X/max(X));
Y=PSDout.Data;
PYout=10*log10(Y/max(Y));
f=PSDout.Frequencies;

%% Iterations
step=15e3; % numel(y) 10e3 % number of samples for each iteration in indirect learning
FinalTest.Enable='off'; % test the best PD model with signal of number of FinalTest.Step
FinalTest.Step=15000;
N=length(PAin)-rem(length(PAin),step);
clear Data
Data.Fs=Fs;
Data.ACPR=ACPR;

Stimulus.SelfEvaluate='off';
Stimulus.wf=Synchronizer.InputRef.IQdata;
Stimulus.Fs=Fs;
Stimulus.ACPR=ACPR;
Stimulus.EvalStart=1;
if strcmpi(Stimulus.SelfEvaluate,'on')
    Stimulus.EvalStep=step;%15000 1500
else
    Stimulus.EvalStep=15000;
end
Stimulus.EvalStop=Stimulus.EvalStart+Stimulus.EvalStep-1;

%% Model Structure
ModelIdentifier.Model.Type='gmp';

ModelIdentifier.Model.NL(1).Kv=[0:10]; %0:9
ModelIdentifier.Model.Mem(1).Lv=[0:1];%0:9;

if strcmpi(ModelIdentifier.Model.Type,'gmp')
    ModelIdentifier.Model.NL(2).Kv=[1];
    ModelIdentifier.Model.Mem(2).Lv=[0:3];%0:7;
    ModelIdentifier.Model.Mem(2).Mv=[1];
    
    ModelIdentifier.Model.NL(3).Kv=[1:3];
    ModelIdentifier.Model.Mem(3).Lv=[0:2];%0:7;
    ModelIdentifier.Model.Mem(3).Mv=[1:2];
end
NbCoeff=nbcoefgmp(ModelIdentifier.Model);
ModelIdentifier.Model.Coeff=zeros(NbCoeff,1);
ModelIdentifier.Model.Coeff(1)=1;

%% Identification
start=Stimulus.EvalStart;%((ns-1)*NbIterPerStage+(n-1))*step+1;
stop=start+step-1;
Data.Out=y(start:stop);%Stage(ns+1).x; % the last preceding stage
Data.In=u(start:stop);

ModelIdentifier.Model=gmpidentifier(Data,ModelIdentifier.Model);
ModelPA=ModelIdentifier.Model;
% Evaluation
z=gmp(u(Stimulus.EvalStart:Stimulus.EvalStop),ModelIdentifier.Model);

Result.Est=y(Stimulus.EvalStart:Stimulus.EvalStop);
Result.Meas=z;
%         Result.ACPR=Test(p).ACPR;
Result.Fs=Fs;
%         [NMSEv(p), ACEPRv(p)]=modeval(Result,Hs);
NMSE=nmse(Result);

