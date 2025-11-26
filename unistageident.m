function SystIter=unistageident(Stimulus,PA,ACPRout,Algorithm,PD,...
    Noise,SpectrumObject)
% SystIter=unistageident(Stimulus,PA,ACPRout,Algorithm,PD,...
%    Noise,SpectrumObject)
% This function implements multistage identification algorithms.
% It takes the following input arguments:
% Stimulus		: Structure with  three fields
%					wf	: for waveform, an Nx1 vector, input samples
%						additional code is needed to take the case 1xN)
%					Fs	: sampling frequency
%					ACPR: structure with 2 fields (BW: for bandwidth and Offset)
% PA			: Structure with many fields, two of which are intersting for
% 					this function
%					Model	: Structure of a NL model
%					G		: Linear gain
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
% Noise			: Structure with 2 fields (in this function)
%					Enable	: string ('on','off')
%							to activate or desactivate added wgn in the feedback path
% 					SNR		: Signal to noise ration
% SpectrumObject: Matlab spectrum object to be used with spectrum class
% Output arguments: To be completed later
%
% 	(c) Mazen Abi Hussein, 2013
% 	(c) Siqi WANG, 2018

% Model=PD.Stage(1).Model;
% clear PD
% PD.Model=Model;
ActiveNoise=Noise.Enable;
snr=Noise.SNR;
Hs=SpectrumObject;

CFR=PD.CFR;
PAin=PD.CFR.Out;
us=Stimulus.wf;
Fs=Stimulus.Fs;
ACPR=Stimulus.ACPR;

step=Algorithm.NbSampPerIter;
mu=Algorithm.DampingNewtonFactor;
Data.Fs=Fs;
Data.ACPR=ACPR;


PA.Model=PA.Model;
SystIter(1).ACPR0=ACPRout;
NbIter=Algorithm.NbIterPerStage;
cs=1; % for current identified stage
for n=1:NbIter
    u=PAin((n-1)*step+1:n*step);
    SystIter(n).u=PAin;%u; %#ok<*AGROW>
    pwrdbm(u)
    SystIter(n).us=us((n-1)*step+1:n*step); %#ok<*AGROW>
    % Through all the stages
    % x designates the output of the current stage
    if strcmp(PD.Model.Type,'dvr')
        PD.x=dvr(u,PD.Model);
    else
        PD.x=gmp(u,PD.Model);
    end
    
    % Clipping outliers at PA input
    if strcmp(CFR.Position,'Behind')
%         PAinMax=PA.LimitPD;%max(abs(PAin))/10^(CFR.delta/10);
%         I= abs(PD.x)>PAinMax;
%         PD.x(I)=PD.x(I)./abs(PD.x(I)).*PAinMax;
        CFR.Thres=PA.InSat; %PA.InSat PA.LimitPD
        CFR.In=PD.x;
        CFR=cfrfiltering(CFR);
        PD.x=CFR.Out;
    end
%     I= abs(PD.x)>PA.LimitPD;
%     PD.x(I)=PD.x(I)./abs(PD.x(I)).*PA.LimitPD;
    % Output PA    
    y=wiener(PD.x,PA.Model);
    
    SystIter(n).y=y;
    [SystIter(n).ACPR, PSDy]=acpr(y,Fs,ACPR,Hs);
    SystIter(n).ACPRimpr=acprdiff(ACPRout,SystIter(n).ACPR);
    Y=PSDy.Data;
    % through the preceding stages of the current stage in the feedback
    % loop.
    % z designates the output of a stage in the FB loop
    
    % + noise
    y=y/PA.G;
    if strcmpi(ActiveNoise,'on')
        noise=wgn(size(y,1),size(y,2),pwrdbm(y)-snr,...
            50,'dBm','complex');
        y=y+noise;
    end
    
    %         for s=1:-1:cs+1
    %             PD.Stage(s).z=mp(y,PD.Stage(s).Model);
    %             y=PD.Stage(s).z;
    %         end
    Data.In=y; % the last preceding stage
    Data.Out=PD.x; % PAin(1:step) 
    if strcmp(PD.Model.Type,'dvr')
        PD.Model=dvridentifier(Data,PD.Model);
    else
        PD.Model=gmpidentifier(Data,PD.Model);
    end
    
    if n>1
        PD.Model.Coeff=(1-mu)*SystIter(n-1).PD.Model.Coeff+mu*PD.Model.Coeff;
    end
    
    %% Test DPD effect    
    if strcmp(PD.Model.Type,'dvr')
        tmp=dvr(PAin,PD.Model);
    else
        tmp=gmp(PAin,PD.Model);
    end
    if strcmp(CFR.Position,'Behind')
        CFR.Thres=PA.InSat; %PA.InSat PA.LimitPD
        CFR.In=tmp;
        CFR=cfrfiltering(CFR);
        tmp=CFR.Out;
    end
    SystIter(n).x=tmp;
    y=wiener(tmp,PA.Model);
    %clipping outliers
%     I= abs(y)>max(abs(u))*PA.G;
%     y(I)=y(I)./abs(y(I)).*abs(u(I))*PA.G;
    %See the performance of identified model
    % + noise
    if strcmpi(ActiveNoise,'on')
        noise=wgn(size(y,1),size(y,2),pwrdbm(y)-snr,...
            50,'dBm','complex');
        y=y+noise;
    end
    SystIter(n).y=y;
    [SystIter(n).ACPR, PSDy]=acpr(y,Fs,ACPR,Hs);
    SystIter(n).ACPRimpr=acprdiff(ACPRout,SystIter(n).ACPR);
    Y=PSDy.Data;
    
    SystIter(n).PY=10*log10(Y/max(Y));
    SystIter(n).PD=PD;
    SystIter(n).CFR=CFR;
    SystIter(n).StageIterNb=n;
    SystIter(n).StageNb=1;
%     SystIter(n).NbOutliers=length(I);
end

%% Reveal complexity
if isfield(CFR,'Complexity')
    fprintf('\n\t CFR.Complexity: %.2s', CFR.Complexity)
end