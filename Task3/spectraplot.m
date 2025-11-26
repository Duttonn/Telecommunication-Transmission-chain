function spectraplot(x,Fs)


    SegLength=2^10;
    Window=blackman(SegLength);%'blackman';
    overlap=50;

[X,f] = pwelch(x,Window,overlap,[],Fs,'center');

h=figure('DefaultAxesFontSize',20,'WindowStyle','docked','NumberTitle','off');
set(h,'Name','Spectra Comparison')
plot(f*1e-6, 10*log10(X));

end