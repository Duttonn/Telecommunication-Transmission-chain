function PdBm = pwrdbm(x,R)
%
% PdBm = pwrdbm(x,R)
% This function computes the power in dBm of any signal x
% x: time domain sequence
% R: Resistance (optional, default 50 Ohms).
%
% (c) Mazen Abi Hussein, 2008
%

if nargin<2
    R=50;
end
PdBm=10*log10(mean(abs(x).^2)/(R*1e-3));

