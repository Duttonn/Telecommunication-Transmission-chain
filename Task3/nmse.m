function NMSE=nmse(Data,Options)
% 
% [NMSE]=nmse(Data)
% This function evaluates the identified model in terms of the NMSE.
% Data.Meas : the measured (reference) signal.
% Data.Est  : estimated (model output) signal.
% Options   : optional argument of type structure for additional options
%           Filed 1: "Normalize", a string that can be 'on' or 'off' to
%           activate or desactivate normalization, respectively. Default
%           value is 'on'. 
%
% Output arguments:
% NMSE      : Normalized Mean-Square Error.
%           Structure with three fields, I, Q and Mag to compute the
%           nmse separetly on I, Q and Magnitude.
%
% (c) Mazen Abi Hussein, 2008
% maj 131029 <OV>
% maj 140516 <MAH> - Changing output argument to structure to compute the
% nmse separetly on I, Q and Magnitude, and adding an input argument
% Options, mainly to control normalization.
%

if nargin<2
    Options.Normalize='on';
end

ymeas=Data.Meas;
yest=Data.Est;

if strcmpi(Options.Normalize,'on')
    yest=fixpwr(yest,pwrdbm(ymeas));
end

Imeas=real(ymeas);
Qmeas=imag(ymeas);
Mmeas=abs(ymeas);

Iest=real(yest);
Qest=imag(yest);
Mest=abs(yest);


NMSE.I=10*log10(sum(abs(Imeas-Iest).^2)/sum(abs(Imeas).^2));
if sum(abs(Qmeas).^2)==0
    NMSE.Q=-inf;
else
    NMSE.Q=10*log10(sum(abs(Qmeas-Qest).^2)/sum(abs(Qmeas).^2));
end
NMSE.Mag=10*log10(sum(abs(Mmeas-Mest).^2)/sum(abs(Mmeas).^2));
NMSE.Z=10*log10(sum(abs(ymeas-yest).^2)/sum(abs(ymeas).^2));



