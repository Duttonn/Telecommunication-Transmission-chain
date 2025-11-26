function y2=timealign(x,y)
% TIMEALIGN - function to align one signal to another
% 
% y2=timealign(x,y) shifts the sampling instants of y such that x and y2
% are aligned, to a subsample resolution. 
% (c) Thomas Eriksson 2015

%N=20000;

% First: find integer alignment
%r=xcorr(x,y,N);
r=xcorr(x,y);
N=(length(r)-1)/2;
[~,maxind]=max(r);
adjust=N+1-maxind;

% Then: find accurate alignment to sub-sample resolution
tdelta=fminbnd(@(tdelta) -abs(circdelay_local(r,tdelta-adjust,N+1)), -0.5, 0.5);  %,optimset('TolX',1e-12) if better accuracy needed

% Do the time alignment
y2=circdelay_local(y,adjust-tdelta);

% Do the phase alignment
y2=y2.*exp(1i*angle(y2'*x));

if isreal(y)
    y2=real(y2);
end


function x2=circdelay_local(x,delay,N) 
% time shifting by a linear phase addition in the spectral domain.
% Parameter N is only used in the optimization, otherwise unnessesary

x2=ifft(ifftshift(fftshift(fft(x)).*exp(1i*2*pi*delay*(-length(x)/2:length(x)/2-1)'/length(x))));
if nargin>2
    x2=x2(N); % this is just for the optimization fminbnd
end
