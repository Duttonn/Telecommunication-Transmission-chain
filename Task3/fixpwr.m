
function xfp =fixpwr(x,PdBm,R)
% xfp =fixpwr(x,PdBm,R)
% This function takes a signal as an argument and fix its power in dBm to
% the specified level PdBm.
% x     : time domain sequence.
% PdBm  : the power of the output signal.
% xfp   : the ouput signal.
% R     : Resistance (default, 50 Ohms).
%
% (c) Mazen Abi Hussein, 2008


%% Method 1
% First fix the pwr to 0dBm by multiplying the input signal by a factor
% So we must compute this factor:
% 10log10(Plin/1e-3)=0 => Plin/1e-3=1  with Plin= E[|A*x(t)|²]/R; 
% We compute A as A=sqrt(1e-3/Plin). Then we must multiply the signal with
% another factor A1 in order to have 10*log10(A1²)=PdBm
% PdBm = 10*log10(E[|A1*x(t)|^2]/(R*1e-3))=10*log10(A1^2)+0dBm


% Plin=mean(abs(x).^2)/R;
% A=sqrt(1e-3/Plin);
% x=A*x;% normalized
% A1=sqrt(10^(P_dBm/10));
% xfp=A1*x;
% % 10*log10(mean(abs(xfp).^2)/(R*.001))

%% Second method easier to understand
% We want to multiply the signal by a factor A in order to have an average
% pwr in dBm equal to PdBm
% ==> PdBm= 10*log10(E[|A1*x(t)|²]/(R*1e-3))=10log10(A²*Plin/1e-3)

if nargin<3
    R=50;
end
Plin=mean(abs(x).^2)/(R);

A=sqrt(10^((PdBm-30)/10)/Plin);

xfp=A*x;
% 
% 10*log10(mean(abs(xfp).^2)/(R*.001))