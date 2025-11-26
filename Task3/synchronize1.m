function Synchronizer=synchronize1(Synchronizer,DisplayOptions)
% [Synchronizer]=synchronize(Synchronizer,<DisplayOptions>)
% This function takes as arguments two unsynchronized signals x1 and x2,
% and use the first one as a refernce signal to synchronize the second one.
% The two signals have the same sampling frequency Fs and same bandwidth B.
% The output of this function is the synchronized version of the second
% signal, x2sync.
% To be generalized later to synchronize two signals with different
% sampling frequencies.
% coefs: are the estimated coefs for the phase shift coefs)[a c]: 2.pi.f.a+c
%
% (c) Mazen Abi Hussein, 2008
% maj: O.Venard 19/09/2012, O.Venard 06/06/2013
% maj: M.Abi Hussein 16/05/2014
% Changing input/output arguments to be more compliant with object oriented
% structures - backward compatibility: No

% Intercorrelation function and FcFs correction are added by Siqi WANG in
% 04/2017

if nargin<2
    DisplayOptions.InterCorrArg.Enable='off';
    DisplayOptions.IQ.Enable='off';
end

if ~isfield(Synchronizer,'EnableCoeffEst')
    Synchronizer.EnableCoeffEst='on';
end
if ~isfield(Synchronizer,'FineSync') %added by Siqi
    Synchronizer.FineSync.Failed=0;
end
if ~isfield(Synchronizer.FineSync,'Failed') %added by Siqi
    Synchronizer.FineSync.Failed=0;
end
if ~isfield(Synchronizer.FineSync,'NbDiv') %added by Siqi
    Synchronizer.FineSync.NbDiv=5;
end
if ~isfield(Synchronizer,'Filter') %added by Siqi
    Synchronizer.Filter.Enable='on';
end
if ~isfield(Synchronizer,'PhaseCorrection') %added by Siqi
    Synchronizer.PhaseCorrection.Enable='off';
end
if ~isfield(DisplayOptions,'InterCorrArg')
    DisplayOptions.InterCorrArg.Enable='off';
end
if ~isfield(DisplayOptions,'IQ')
    DisplayOptions.IQ.Enable='off';
end
if ~isfield(DisplayOptions.IQ,'Offset')
    DisplayOptions.IQ.Offset=1;
end
if ~isfield(DisplayOptions.IQ,'Scope')
    DisplayOptions.IQ.Scope=1000;
end

% After synchronization we may want to remove samples at borders
% This optional feature allows to remove a percentage of samples from the
% beginning and the end of the signal
% It may be not necessary/ under investigation
if ~isfield(Synchronizer,'TruncateSignals')
    Synchronizer.TruncateSignals.Enable='off';
elseif ~isfield(Synchronizer.TruncateSignals,'Perc')
    Synchronizer.TruncateSignals.Perc=2;
end

if ~isfield(Synchronizer,'NMSE')
    Synchronizer.NMSE.Threshold=-20;
    Synchronizer.NMSE.ComputationMethod='FullRange';
    % 'FullRange' 'SmallestRange' 'STDRangeMean' 'STDRangeZero'
else
    if ~isfield(Synchronizer.NMSE,'Threshold')
        Synchronizer.NMSE.Threshold=-20;
    end
    if ~isfield(Synchronizer.NMSE,'ComputationMethod')
        Synchronizer.NMSE.ComputationMethod='FullRange';
    elseif strcmpi(Synchronizer.NMSE.ComputationMethod,'STDRangeMean') ||...
            strcmpi(Synchronizer.NMSE.ComputationMethod,'STDRangeZero')
        if ~isfield(Synchronizer.NMSE,'RangeStd')
            Synchronizer.NMSE.RangeStd=2;
        end
    end
end


if strcmpi(Synchronizer.Method, 'intercorrelation')
    Synchronizer=intercorrelation(Synchronizer,DisplayOptions);
else
    y=Synchronizer.Input.IQdata;
    %% Elimination of PA nonlinearity influence
    if strcmp(Synchronizer.Filter.Enable,'on')
        [ynew,L] = datalowpassfilter(y);
        %% Delay Compensation Method 2
        
        [acor,lag] = xcorr(y,ynew);
        [~,Iv] = max(abs(acor));
        
        lagDiff = lag(Iv);
%         timeDiff = lagDiff/Fs
        % figure
        % plot(lag,abs(acor))
        yfd1=circshift(ynew,lagDiff); %time shift correction
    else
        yfd1=y;
    end
    Synchronizer.Input.IQdata=yfd1;
    Synchronizer.EnableCoeffEst='on';
    Synchronizer=conventional(Synchronizer,DisplayOptions);
    Synchronizer.Input.IQdata=y;
    Synchronizer.EnableCoeffEst='on'; %off
    Synchronizer=conventional(Synchronizer,DisplayOptions);
end
    %% FcFs correction
SystUT2.Input=Synchronizer.InputRef;
SystUT2.Output=Synchronizer.Output;
SystUT2.NMSE=Synchronizer.NMSE;
SystUT2=frequencycorrection(SystUT2); %added by Siqi
Synchronizer.Output=SystUT2.Output;
if isfield(SystUT2,'Fc')
    disp(SystUT2.Fc.Delta)
end
if isfield(SystUT2,'Fs')
    disp(SystUT2.Fs.Delta)
end
end

function Synchronizer=intercorrelation(Synchronizer,DisplayOptions)
x=Synchronizer.InputRef.IQdata;
y=Synchronizer.Input.IQdata;

[acor,lag] = xcorr(x,y);
[~,Iv] = max(abs(acor));%abs(acor)

lagDiff = lag(Iv);
fprintf(['Intercorrelation result: lag=',num2str(lagDiff)])
Synchronizer.Output=Synchronizer.Input;
Synchronizer.Output.IQdata=circshift(y,[lagDiff,0]);
Len=length(y);
% if lagDiff<0
%     sector=1:Len+lagDiff-1;
% else
%     sector=lagDiff+1:Len;
% end
% x=x(sector);
y=Synchronizer.Output.IQdata;%(sector);

Data.Est=Synchronizer.InputRef.IQdata;%(sector);
Data.Meas=fixpwr(Synchronizer.Output.IQdata,pwrdbm(Data.Est));
Synchronizer.NMSE.Value=nmse(Data);

if strcmpi(DisplayOptions.NMSE.Enable,'on')
    fprintf(['\n NMSE between synchronized and original before ',...
        'Phase shift correction:',...
        '\n\t I (dB)  : %.2f.',...
        '\n\t Q (dB)  : %.2f.',...
        '\n\t Mag (dB): %.2f.',...
        '\n\t Z (dB): %.2f.\n'],Synchronizer.NMSE.Value.I,...
        Synchronizer.NMSE.Value.Q,Synchronizer.NMSE.Value.Mag...
        ,Synchronizer.NMSE.Value.Z)
end
%% Phase & Frequency Correction
if strcmpi(Synchronizer.PhaseCorrection.Enable,'on')
    if ~isfield(Synchronizer.PhaseCorrection,'Method')
        Synchronizer.PhaseCorrection.Method='LS';
    end
    
    t=(1:length(x))/Synchronizer.Input.Fs;
    t=t(:);
    G=x.*conj(y);
    xv1= polyfit(t,angle(G),1);
    Phi.Delta=(xv1(1)*t+xv1(2));
    
    Synchronizer.Output.IQdata=Synchronizer.Output.IQdata.*exp(1i*Phi.Delta);
    Data.Meas=Synchronizer.InputRef.IQdata;
    Data.Est=Synchronizer.Output.IQdata;
    Synchronizer.NMSE.Value=nmse(Data);
    if strcmpi(DisplayOptions.NMSE.Enable,'on')
    fprintf(['\n NMSE between synchronized and original with ',...
        'Phase shift & carrier frequency correction:',...
        '\n\t I (dB)  : %.2f.',...
        '\n\t Q (dB)  : %.2f.',...
        '\n\t Mag (dB): %.2f.',...
        '\n\t Z (dB): %.2f.\n'],Synchronizer.NMSE.Value.I,...
        Synchronizer.NMSE.Value.Q,Synchronizer.NMSE.Value.Mag...
        ,Synchronizer.NMSE.Value.Z)
    end
end

Data.Meas=Synchronizer.InputRef.IQdata;
Data.Est=Synchronizer.Output.IQdata;
Synchronizer.NMSE.Value=nmse(Data);
if Synchronizer.NMSE.Value.Mag>Synchronizer.NMSE.Threshold || ...
        Synchronizer.NMSE.Value.I>Synchronizer.NMSE.Threshold || ...
        Synchronizer.NMSE.Value.Q>Synchronizer.NMSE.Threshold
    warning(['Bad synchronization, NMSE (I=',...
        num2str(Synchronizer.NMSE.Value.I),...
        ', | Q=',num2str(Synchronizer.NMSE.Value.Q),...
        ', | Mag=',num2str(Synchronizer.NMSE.Value.Mag),...
        ') > Threshold (=',num2str(Synchronizer.NMSE.Threshold),')'])
    Synchronizer.Failed=1;
    Synchronizer.FineSync.Failed=1; %added by Siqi
else
    Synchronizer.Failed=0;
    Synchronizer.FineSync.Failed=0; %added by Siqi
end
Synchronizer.TruncInputRef=Synchronizer.InputRef;
Synchronizer.TruncOutput=Synchronizer.Output;
end

function Synchronizer=conventional(Synchronizer,DisplayOptions)
x1=Synchronizer.InputRef.IQdata;
x2=Synchronizer.Input.IQdata;
B=Synchronizer.InputRef.BW;
Fs=Synchronizer.InputRef.Fs;
%% FFT of x1 & x2
x1=fixpwr(x1,pwrdbm(x2));
nfft=numel(x1);
X1=fft(x1);
X1=fftshift(X1);
X2=fft(x2);
X2=fftshift(X2);
f=(-nfft/2:1:nfft/2-1)*Fs/nfft;
f=f(:);

%% FT of the autocorrelation function & Phase estimation
Rx1x2=X1.*conj(X2);
ArgRx1x2 = angle(Rx1x2);

% Line equation: y=ax+c => Ax=b, where x here is the vector of unknown
% coefficients a and c, x^T=[a c]; b=yv; and A=[fv 1's];
% pts=find(f>=-(B/2)*0.8 & f<=(B/2)*0.8);
% pts=find(f>=-(B/2)*0.8 & f<=0); % It doesn't work when the signal inside
% BW is broken
pts=find(f>=0 & f<=(B/2)*0.8);

if strcmp(Synchronizer.EnableCoeffEst,'on')
    fv=f(pts);
    yv=unwrap(ArgRx1x2(pts));
    xv= polyfit(fv,yv,1);
else
    xv=Synchronizer.Coeff;
end

ArgRx1x2_est=xv(1)*f + xv(2);
X2c=X2.*exp(1i*(ArgRx1x2_est));

X2c=ifftshift(X2c);
x2sync=ifft(X2c);

% disp(xv)
coefs=[xv(1) xv(2)];

Synchronizer.Output=Synchronizer.Input;
Synchronizer.Output.IQdata=x2sync;

Synchronizer.TruncInputRef=Synchronizer.InputRef;
Synchronizer.TruncInput=Synchronizer.Input;
Synchronizer.TruncOutput=Synchronizer.Output;
Synchronizer.Coeff=coefs;

if strcmpi(Synchronizer.TruncateSignals.Enable,'on')
    ExtraSamp=(Synchronizer.TruncateSignals.Perc/100)*Synchronizer.Input.Length;
    ExtraSamp=fix(ExtraSamp);
    Synchronizer.TruncOutput.IQdata=x2sync(ExtraSamp+1:end-ExtraSamp);
    Synchronizer.TruncOutput.Length=numel(Synchronizer.TruncOutput.IQdata);
    InRef=Synchronizer.InputRef.IQdata(ExtraSamp+1:end-ExtraSamp);
    Synchronizer.TruncInputRef.IQdata=InRef;
    Synchronizer.TruncInputRef.Length=numel(InRef);
    InSync=Synchronizer.Input.IQdata(ExtraSamp+1:end-ExtraSamp);
    Synchronizer.TruncInput.IQdata=InSync;
    Synchronizer.TruncInput.Length=numel(InSync);
    %% NMSE
    x=Synchronizer.TruncInputRef.IQdata;
    y=Synchronizer.TruncOutput.IQdata;
    y=fixpwr(y,pwrdbm(x));
    if strcmpi(Synchronizer.NMSE.ComputationMethod,'FullRange')
        Iv=1:numel(y);
    elseif strcmpi(Synchronizer.NMSE.ComputationMethod,'SmallestRange')
        if max(abs(x))>max(abs(y))
            Iv=abs(x)<max(abs(y));
        else
            Iv=abs(y)<max(abs(x));
        end
    elseif strcmpi(Synchronizer.NMSE.ComputationMethod,'STDRangeMean')
        stdy=std(abs(y));
        Iv=abs(y)<mean(abs(y))+Synchronizer.NMSE.RangeStd*stdy/2 &...
            abs(y)>mean(abs(y))-Synchronizer.NMSE.RangeStd*stdy/2;
    elseif strcmpi(Synchronizer.NMSE.ComputationMethod,'STDRangeZero')
        stdy=std(abs(y));
        Iv=abs(y)<Synchronizer.NMSE.RangeStd*stdy;
    end
    Data.Meas=x(Iv);
    Data.Est=y(Iv);
    Synchronizer.NMSE.Value=nmse(Data);
else
    %% NMSE
    x=Synchronizer.InputRef.IQdata;
    y=Synchronizer.Output.IQdata;
    y=fixpwr(y,pwrdbm(x));
    if strcmpi(Synchronizer.NMSE.ComputationMethod,'FullRange')
        Iv=1:numel(y);
    elseif strcmpi(Synchronizer.NMSE.ComputationMethod,'SmallestRange')
        if max(abs(x))>max(abs(y))
            Iv=abs(x)<max(abs(y));
        else
            Iv=abs(y)<max(abs(x));
        end
    elseif strcmpi(Synchronizer.NMSE.ComputationMethod,'STDRangeMean')
        stdy=std(abs(y));
        Iv=abs(y)<mean(abs(y))+Synchronizer.NMSE.RangeStd*stdy/2 &...
            abs(y)>mean(abs(y))-Synchronizer.NMSE.RangeStd*stdy/2;
    elseif strcmpi(Synchronizer.NMSE.ComputationMethod,'STDRangeZero')
        stdy=std(abs(y));
        Iv=abs(y)<Synchronizer.NMSE.RangeStd*stdy;
    end
    Data.Meas=x(Iv);
    Data.Est=y(Iv);
    Synchronizer.NMSE.Value=nmse(Data);
    Synchronizer.TruncInputRef=Synchronizer.InputRef;%added by Siqi
    Synchronizer.TruncInput=Synchronizer.Input;%added by Siqi
    Synchronizer.TruncOutput=Synchronizer.Output;%added by Siqi
end


if Synchronizer.NMSE.Value.Mag>Synchronizer.NMSE.Threshold || ...
        Synchronizer.NMSE.Value.I>Synchronizer.NMSE.Threshold || ...
        Synchronizer.NMSE.Value.Q>Synchronizer.NMSE.Threshold
    warning(['Bad synchronization, NMSE (I=',...
        num2str(Synchronizer.NMSE.Value.I),...
        ', | Q=',num2str(Synchronizer.NMSE.Value.Q),...
        ', | Mag=',num2str(Synchronizer.NMSE.Value.Mag),...
        ') > Threshold (=',num2str(Synchronizer.NMSE.Threshold),')'])
    Synchronizer.Failed=1;
else
    Synchronizer.Failed=0;
end


%% Subdividing the whole bandwidth into half BWs, B/10
while Synchronizer.Failed==1 && strcmpi(Synchronizer.FineSync.Enable,'on') %if
    warning('Fine synchronization: BW=B/10')
    if Synchronizer.FineSync.Failed==1 %added by Siqi
        NbDiv=500;
    else
        NbDiv=Synchronizer.FineSync.NbDiv;
    end
    div=0:B/2/NbDiv:(B/2)*.8;%0:B/2/5:(B/2)*.8;
    av=zeros(numel(div)-1,1);
    bv=av;
    Diffv=av;
    if strcmp(Synchronizer.EnableCoeffEst,'on')
        for i=1:numel(div)-1
            % Line equation: y=ax+c => Ax=b, where x here is the vector of unknown
            % coefficients a and c, x^T=[a c]; b=yv; and A=[fv 1's];
            pts=find(f>=div(i) & f<=div(i+1));
            fv=f(pts);
            yv=ArgRx1x2(pts);
            %     [~,av(i),bv(i)] = regression(fv,yv);
            p= polyfit(fv,yv,1);
            av(i)=p(1);
            bv(i)=p(2);
            ArgRx1x2_est=av(i)*f+bv(i);
            X2c=X2.*exp(1i*(ArgRx1x2_est));
            X2c=fftshift(X2c);
            x2sync=ifft(X2c);
            %     Diffv(i) = norm(real(x2sync) - real(x1))+...
            %         norm(imag(x2sync) - imag(x1))...
            %         + norm(abs(x2sync) - abs(x1));
            Diffv(i) = norm(real(x2sync) - real(x1))+...
                norm(imag(x2sync) - imag(x1));
            %     Diffv(i) = norm(abs(x2sync) - abs(x1));
            %     Diffv(i) = sqrt(norm(real(x2sync) - real(x1))^2+...
            %         norm(imag(x2sync) - imag(x1))^2);
            
        end
        [min_diff,index]=min(Diffv); %#ok<ASGLU>
    else
        index=1;
        av(index)=Synchronizer.Coeff(1);
        bv(index)=Synchronizer.Coeff(2);
    end
    ArgRx1x2_est=av(index)*f + bv(index);
    X2c=X2.*exp(1i*(ArgRx1x2_est));
    X2c=fftshift(X2c);
    x2sync=ifft(X2c);
    
    %     disp([av(index) bv(index)])
    coefs=[av(index) bv(index)];
    
    Synchronizer.Output=Synchronizer.Input;
    Synchronizer.Output.IQdata=x2sync;
    
    Synchronizer.TruncInputRef=Synchronizer.Input;
    Synchronizer.TruncInput=Synchronizer.Input;
    Synchronizer.TruncOutput=Synchronizer.Input;
    Synchronizer.Coeff=coefs;
    if strcmpi(Synchronizer.TruncateSignals.Enable,'on')
        ExtraSamp=(Synchronizer.TruncateSignals.Perc/100)*Synchronizer.Input.Length;
        ExtraSamp=fix(ExtraSamp);
        Synchronizer.TruncOutput.IQdata=x2sync(ExtraSamp+1:end-ExtraSamp);
        Synchronizer.TruncOutput.Length=numel(Synchronizer.TruncOutput.IQdata);
        InRef=Synchronizer.InputRef.IQdata(ExtraSamp+1:end-ExtraSamp);
        Synchronizer.TruncInputRef.IQdata=InRef;
        Synchronizer.TruncInputRef.Length=numel(InRef);
        InSync=Synchronizer.Input.IQdata(ExtraSamp+1:end-ExtraSamp);
        Synchronizer.TruncInput.IQdata=InSync;
        Synchronizer.TruncInput.Length=numel(InSync);
        %% NMSE
        x=Synchronizer.TruncInputRef.IQdata;
        y=Synchronizer.TruncOutput.IQdata;
        y=fixpwr(y,pwrdbm(x));
        if strcmpi(Synchronizer.NMSE.ComputationMethod,'FullRange')
            Iv=1:numel(y);
        elseif strcmpi(Synchronizer.NMSE.ComputationMethod,'SmallestRange')
            if max(abs(x))>max(abs(y))
                Iv=abs(x)<max(abs(y));
            else
                Iv=abs(y)<max(abs(x));
            end
        elseif strcmpi(Synchronizer.NMSE.ComputationMethod,'STDRangeMean')
            stdy=std(abs(y));
            Iv=abs(y)<mean(abs(y))+Synchronizer.NMSE.RangeStd*stdy/2 &...
                abs(y)>mean(abs(y))-Synchronizer.NMSE.RangeStd*stdy/2;
        elseif strcmpi(Synchronizer.NMSE.ComputationMethod,'STDRangeZero')
            stdy=std(abs(y));
            Iv=abs(y)<Synchronizer.NMSE.RangeStd*stdy;
        end
        Data.Meas=x(Iv);
        Data.Est=y(Iv);
        Synchronizer.NMSE.Value=nmse(Data);
    else
        %% NMSE
        x=Synchronizer.InputRef.IQdata;
        y=Synchronizer.Output.IQdata;
        y=fixpwr(y,pwrdbm(x));
        if strcmpi(Synchronizer.NMSE.ComputationMethod,'FullRange')
            Iv=1:numel(y);
        elseif strcmpi(Synchronizer.NMSE.ComputationMethod,'SmallestRange')
            if max(abs(x))>max(abs(y))
                Iv=abs(x)<max(abs(y));
            else
                Iv=abs(y)<max(abs(x));
            end
        elseif strcmpi(Synchronizer.NMSE.ComputationMethod,'STDRangeMean')
            stdy=std(abs(y));
            Iv=abs(y)<mean(abs(y))+Synchronizer.NMSE.RangeStd*stdy/2 &...
                abs(y)>mean(abs(y))-Synchronizer.NMSE.RangeStd*stdy/2;
        elseif strcmpi(Synchronizer.NMSE.ComputationMethod,'STDRangeZero')
            stdy=std(abs(y));
            Iv=abs(y)<Synchronizer.NMSE.RangeStd*stdy;
        end
        Data.Meas=x(Iv);
        Data.Est=y(Iv);
        Synchronizer.NMSE.Value=nmse(Data);
        Synchronizer.TruncInputRef=Synchronizer.InputRef;%added by Siqi
        Synchronizer.TruncInput=Synchronizer.Input;%added by Siqi
        Synchronizer.TruncOutput=Synchronizer.Output;%added by Siqi
    end
    
    if Synchronizer.NMSE.Value.Mag>Synchronizer.NMSE.Threshold || ...
            Synchronizer.NMSE.Value.I>Synchronizer.NMSE.Threshold || ...
            Synchronizer.NMSE.Value.Q>Synchronizer.NMSE.Threshold
        warning(['Bad synchronization, NMSE (I=',...
            num2str(Synchronizer.NMSE.Value.I),...
            ', | Q=',num2str(Synchronizer.NMSE.Value.Q),...
            ', | Mag=',num2str(Synchronizer.NMSE.Value.Mag),...
            ') > Threshold (=',num2str(Synchronizer.NMSE.Threshold),')'])
        Synchronizer.Failed=1;
        Synchronizer.FineSync.Failed=1; %added by Siqi
    else
        Synchronizer.Failed=0;
        Synchronizer.FineSync.Failed=0; %added by Siqi
    end
    
end

if strcmpi(DisplayOptions.NMSE.Enable,'on')
    if exist('NbDiv','var')
        fprintf(['\n NMSE between synchronized and original ',...
            'after fine synchronization by %.2f division:',...
            '\n\t I (dB)  : %.2f.',...
            '\n\t Q (dB)  : %.2f.'...
            '\n\t Mag (dB)  : %.2f.'...
            '\n\t Z (dB): %.2f.\n'],NbDiv,Synchronizer.NMSE.Value.I,...
            Synchronizer.NMSE.Value.Q,Synchronizer.NMSE.Value.Mag,...
            Synchronizer.NMSE.Value.Z)
    else
        fprintf(['\n NMSE between synchronized and original:',...
            '\n\t I (dB)  : %.2f.',...
            '\n\t Q (dB)  : %.2f.'...
            '\n\t Mag (dB)  : %.2f.'...
            '\n\t Z (dB): %.2f.\n'],Synchronizer.NMSE.Value.I,...
            Synchronizer.NMSE.Value.Q,Synchronizer.NMSE.Value.Mag,...
            Synchronizer.NMSE.Value.Z)
    end
end
%% Time domain

if (strcmp(DisplayOptions.IQ.Enable,'on'))
    offset=DisplayOptions.IQ.Offset;
    scope=DisplayOptions.IQ.Scope;
    
    Signal(1).Name='RefIn';
    Signal(1).IQdata=Synchronizer.InputRef.IQdata(offset:scope);
    Signal(1).Fs=Synchronizer.InputRef.Fs;
    
    Signal(2).Name='BeforeSync';
    Signal(2).IQdata=Synchronizer.Input.IQdata(offset:scope);
    Signal(2).Fs=Synchronizer.Input.Fs;
    
    Signal(3).Name='Sync';
    Signal(3).IQdata=Synchronizer.Output.IQdata(offset:scope);
    Signal(3).Fs=Synchronizer.Output.Fs;
    
    DispOpt.TimeUnit='us';
    DispOpt.FigureName='I/Q TimeWFsync';
    DispOpt.Normalize='on';
    figureiq(Signal,DispOpt);
end

%% Intercorrelation argument
if (strcmp(DisplayOptions.InterCorrArg.Enable,'on'))
    figure('Name','ArgInterCorr','WindowStyle','docked','NumberTitle','off')
    plot(f*1e-6,ArgRx1x2,f*1e-6,ArgRx1x2_est)
    xlabel('Frequency (MHz)')
    ylabel('Intercorrelation Argument (rad)')
    legend('Actual','Estimated')
    grid on
end
end

