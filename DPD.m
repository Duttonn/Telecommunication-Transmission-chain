function SystIter=DPD(Stimulus,ACPRout,Algorithm,PD)
% SystIter=DPD(Stimulus,ACPRout,Algorithm,PD)
% This function implements DPD algorithms.
% It takes the following input arguments:
% Stimulus		: Structure with  three fields
%					wf	: for waveform, an Nx1 vector, input samples
%						additional code is needed to take the case 1xN)
%					Fs	: sampling frequency
%					ACPR: structure with 2 fields (BW: for bandwidth and Offset)
% ACPRout		: Structure with ACPR measurements of PA output; To avoid
%				computing such measurement inside this function, and used
%				as a refernces to compute ACPR improvement after predistortion
%				(Can be implemented in a better way / to be modified later)
% Algorithm		: Structure with three fields (in this function)
%                   Method : String to choose the method to use:
%                           'Backward' for the first algorithm.
%                           'Alternate' for the second one.
%					NbIter	or NbIterPerStage : Number of iteration
%                           (positive integer, verification
%							should be added later)
%                           NbIterPerStage when using 'Backward' method
%							NbIter when using 'Alternate' method
%					NbSampPerIter	: Number of samples to be used for each system
%										level iteration (positive integer, verification needed)
% PD			: Structure of a general parallel structure nonlinear model,
%				Each stage has its own number of iterations
%
% 	(c) Siqi WANG, 2019

RMSin=Stimulus.RMSin;
PAin=fixpwr(PD.In,RMSin);
Fs=Stimulus.Fs;
ACPR=Stimulus.ACPR;

clear PD.Model

% configure your DPD model parameters here
        PD.Model.NL(1).Kv=[0:6]; 
        PD.Model.Mem(1).Lv=[0:3];
        PD.Model.NL(2).Kv=[1:4];
        PD.Model.Mem(2).Lv=[0:3];
        PD.Model.Mem(2).Mv=[1:2];
        PD.Model.NL(3).Kv=[1:4];
        PD.Model.Mem(3).Lv=[0:3];
        PD.Model.Mem(3).Mv=[1:2];
    PD.Model.bias='off';
    PD.Model.Symmetric='off';
    PD.Model.Type='gmp';
    NbCoeff=nbcoefgmp(PD.Model);
    PD.Model.Coeff=zeros(NbCoeff,1);
    PD.Model.Coeff(1)=1;

section=Algorithm.NbSampPerIter;
mu=Algorithm.DampingNewtonFactor;
Data.Fs=Fs;
Data.ACPR=ACPR;

SystIter(1).ACPR0=ACPRout;
NbIter=Algorithm.NbIterPerStage;
for n=1:NbIter
    u=PAin(1:section);
    SystIter(n).u=u;
    
    if n>1
            PD.x=gmp(u,PD.Model);
    else
        PD.x=u;
        figure('DefaultAxesFontSize',20,'WindowStyle','docked','NumberTitle','off');
    end
    % Output PA    
[y, RMSout, Idc, Vdc] = RFWebLab_PA_meas_v1_2(PD.x, RMSin);

y=timealign(PD.x,y);

    SystIter(n).y=y;
    [SystIter(n).ACPR, PSDy]=acpr(y,Fs,ACPR);
    SystIter(n).ACPRimpr=acprdiff(ACPRout,SystIter(n).ACPR);
    Y=PSDy.Data;
    SystIter(n).PY=10*log10(Y);
    
    hold on; plot(PSDy.Frequencies*1e-6,10*log10(Y))

    y=fixpwr(y,RMSin);
    
    Data.In=y; 
    Data.Out=PD.x;  
    
        Options.Evaluation.Enable='on';
        [PD.Model,Eval]=gmpidentifier(Data,PD.Model,Options);
    
    %% Test DPD effect    
    
    SystIter(n).Idc=Idc;
    SystIter(n).Vdc=Vdc;
    SystIter(n).RMSout=RMSout;
    SystIter(n).PD=PD;
    SystIter(n).Eval=Eval;
    SystIter(n).StageIterNb=n;
    SystIter(n).StageNb=1;
end
