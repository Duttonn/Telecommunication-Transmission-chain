function SystUT=frequencycorrection(SystUT)
% (c) WANG Siqi, 2015

if ~isfield(SystUT,'CorrectionOption')
    SystUT.CorrectionOption.SamplingFreq.InetrpMethod='linear';
    SystUT.CorrectionOption.start=1;
    SystUT.CorrectionOption.Fs=200e6;
    SystUT.CorrectionOption.NbPts=length(SystUT.Input.IQdata);
end
if ~isfield(SystUT.CorrectionOption,'CorrectionOrder')
    SystUT.CorrectionOption.CorrectionOrder='FsFc'; %FcFs
end
if ~isfield(SystUT.CorrectionOption,'FcCorrection')
    SystUT.CorrectionOption.FcCorrection='on';
end
if ~isfield(SystUT.CorrectionOption,'FsCorrection')
    SystUT.CorrectionOption.FsCorrection='on';
end

if ~isfield(SystUT,'NMSE')
    Data.Est=Input;
    Data.Meas=Output;
    SystUT.NMSE.Value=nmse(Data);
end
%% Fc => Fs Correction
if strcmp(SystUT.CorrectionOption.CorrectionOrder,'FsFc')
    if strcmp(SystUT.CorrectionOption.FsCorrection, 'on')
        SystUT=fscorrection(SystUT);
    end
    if strcmp(SystUT.CorrectionOption.FcCorrection,'on')
        SystUT=fccorrection(SystUT);
    end
else
    %% Fs => Fc Correction
    if strcmp(SystUT.CorrectionOption.FcCorrection,'on')
        SystUT=fccorrection(SystUT);
    end
    if strcmp(SystUT.CorrectionOption.FsCorrection, 'on')
        SystUT=fscorrection(SystUT);
    end
end
end

%% Fs Correction
function SystUT=fscorrection(SystUT)
InterpMethod=SystUT.CorrectionOption.SamplingFreq.InetrpMethod;
start=SystUT.CorrectionOption.start;
nbPts=SystUT.CorrectionOption.NbPts;
stop=start+nbPts-1;
fs=SystUT.Input.Fs;
Input=SystUT.Input.IQdata;
Output=SystUT.Output.IQdata;
OutputCrt=Output;
Fs.Interval.Step=1;
Fs.Interval.Start=-150;%-5
Fs.Interval.Stop=Fs.Interval.Start*(-1);
deltafs=Fs.Interval.Start:Fs.Interval.Step:Fs.Interval.Stop;
% t=(0:stop-1)/Fs;
% t=t(:);
for j=1:length(deltafs)
    Fsp=fs+deltafs(j);
    tp=(0:nbPts-1)/Fsp;
    tp=tp(:);
    ts=(0:nbPts-1)/fs;
    ts=ts(:);
    x=OutputCrt(start:stop);
    OutputCrtFs=interp1(ts,x,tp,InterpMethod,'extrap');
    Data.Est=Input(start:stop);
    Data.Meas=OutputCrtFs;
    NMSE(j)=nmse(Data);
    NMSEZ(j)=NMSE(j).Z;
end

[~, Index]=min(NMSEZ);
Fs.Delta=deltafs(Index);

t=(0:numel(Output)-1)/fs;
t=t(:);
tp=(0:numel(Output)-1)/(fs+Fs.Delta);
tp=tp(:);
OutputCrtFs=interp1(t,Output,tp,InterpMethod,'extrap');

Data.Meas=Input;
Data.Est=OutputCrtFs;
NMSECrt=nmse(Data);
fprintf('NMSE after Fs Correction is:\n')
disp(NMSECrt)
if SystUT.NMSE.Value.Z<NMSECrt.Z
    fprintf('Fs Correction is not needed:\n')
else
    OutputCrt=OutputCrtFs;
    SystUT.NMSE.Value.Z=NMSECrt.Z;
end
SystUT.Fs=Fs;
SystUT.Output.IQdata=OutputCrt;
end
%% Fc Correction
function SystUT=fccorrection(SystUT)
SystUT.FcMethod='LS';
start=SystUT.CorrectionOption.start;
nbPts=SystUT.CorrectionOption.NbPts;
stop=start+nbPts-1;
fs=SystUT.Input.Fs;
Input=SystUT.Input.IQdata;
Output=SystUT.Output.IQdata;
OutputCrt=Output;
if strcmp(SystUT.FcMethod,'LS')
    t=(1:length(Input))/fs;
    t=t(:);
    G=Input.*conj(OutputCrt);
    xv= polyfit(t,angle(G),1);
    Fc.Delta=xv(1)/(2*pi);
    OutputCrtFc=OutputCrt.*exp(1i*(2*pi*Fc.Delta*t+xv(2)));
else
    Fc.Interval.Step=1;%1
    Fc.Interval.Start=-150;%-50
    Fc.Interval.Stop=Fc.Interval.Start*(-1);
    deltafc=Fc.Interval.Start:Fc.Interval.Step:Fc.Interval.Stop;
    t=(start:stop)/fs;
    t=t(:);
    for j=1:length(deltafc)
        OutputCrtFc=OutputCrt(start:stop).*exp(1i*2*pi*deltafc(j)*t);
        Data.Est=Input(start:stop);
        Data.Meas=OutputCrtFc;
        NMSE(j)=nmse(Data);
        NMSEZ(j)=NMSE(j).Z;
    end
    
    [~, Index]=min(NMSEZ);
    Fc.Delta=deltafc(Index);
    t=(0:numel(Output)-1)/fs;
    t=t(:);
    OutputCrtFc=Output.*exp(1i*2*pi*Fc.Delta*t);
end
Data.Meas=Input;
Data.Est=OutputCrtFc;
NMSECrt=nmse(Data);
fprintf('NMSE after phase and Fc Correction is:\n')
disp(NMSECrt)
if SystUT.NMSE.Value.Z<NMSECrt.Z
    fprintf('Phase and Fc Correction is not needed:\n')
else
    OutputCrt=OutputCrtFc;
    SystUT.NMSE.Value.Z=NMSECrt.Z;
end
SystUT.Fc=Fc;
SystUT.Output.IQdata=OutputCrt;
end
