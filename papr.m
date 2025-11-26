
function PAPR=papr(wf,R,type)
% PAPR=papr(x,R,type)
% This function computes the PAPR of the signal
% It takes 3 input arguments:
%   wf   : vector of complex or real values representing a time domain  
%   waveform. It can be an RF modulated signal or its equivalent envelope
%    baseband signal.
%   R    : real, Resistance (default 50 Ohms), optional.
%   type : String, if equal to 'bb' or 'BB', the baseband equivalent PAPR
%   is computed, i.e., -3dB. Useful when using equivalent baseband models.
% It returns one output argument:
%   PARP : real, the PAPR of the signal.
%
% (c) Mazen Abi Hussein, 2008.
%

if nargin<2
    R=50;
end
S=max(abs(wf));

Ppk=10*log10(((S/sqrt(2))^2)/(R*1e-3));

Pav=pwrdbm(wf,R);

PAPR=Ppk-Pav+3; % 3 dB is for the baseband signal

if nargin>2
    if ~strcmpi(type,'bb')
        PAPR=PAPR-3;
    end
end