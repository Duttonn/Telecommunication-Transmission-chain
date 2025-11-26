function [ACPR,PSD] = acpr (x,Fs,ACPR)
% ACPR = acpr (x,Fs,ACPR,R)
% This function computes the Adjacent Channel Power Ratio (ACPR) for any
% time domain signal x.
% x     : time domain signal.
% Fs    : sampling frequency
% ACPR  : structure with two fields (BW and Offset).
% Hs    : The spectral estimation object (Optional).
%
% Output arguments:
% ACPR  : Adjacent Channel Power Ratio, a structure with 6 fields:
% L1 (U1) first lower (upper) adjacent channel, and L2 (U2) second lower 
% (upper) adjacent channel) and the two fields already existing from the
% input argument, Offset and BW.
% PSD   : Power Spectral Density, structure resulting from using psd
% function of matlab on signal x.
%
% (c) Mazen Abi Hussein, 2009 


BW=ACPR.BW;
Offset=ACPR.Offset; 

if nargin==3
%     Window='blackman';
    SegLength=2^10;
    Window=blackman(SegLength);%'blackman';
    overlap=50;
%     Hs = spectrum.welch(Window,SegLength,overlap);
end

[X,f] = pwelch(x,Window,overlap,[],Fs,'center');
% PSD=psd(Hs,x,'Fs',Fs,'CenterDC',true);
% X=PSD.Data;
% f=PSD.Frequencies;
PSD.Data=X;
PSD.Frequencies=f;

% Main channel
MainCH= f>=(-BW/2) & f<=(BW/2);
Pmc=10*log10(sum(X(MainCH)));

% Upper 1
U1CH= f>=(Offset-BW/2)& f<=(Offset+BW/2);
PU1=10*log10(sum(X(U1CH)));

ACPR.U1=PU1-Pmc;

% Lower 1
L1CH= f>=(-Offset-BW/2)& f<=(-Offset+BW/2);
PL1=10*log10(sum(X(L1CH)));

ACPR.L1=PL1-Pmc;

% Upper 2
U2CH= f>=(2*Offset-BW/2)& f<=(2*Offset+BW/2);
PU2=10*log10(sum(X(U2CH)));

ACPR.U2=PU2-Pmc;

% Lower 2
L2CH= f>=(-2*Offset-BW/2)& f<=(-2*Offset+BW/2);
PL2=10*log10(sum(X(L2CH)));

ACPR.L2=PL2-Pmc;
 
