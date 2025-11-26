
LevelUp=2;
frontrun;

load PAinLTE20MHz

N=length(PAin);
ACPR.BW=BW;
ACPR.Offset=1.1*BW;

Gap=100e6;
fc=[-19 0 19]*1e6;
Fs=Fs*numel(fc);
PAin=resample(PAin,numel(fc),1);
% PAin=PAin(1:N);

Nf   = 10;
fp  = BW*0.5; %*(Kav(k)+1)
Ap  = 0.001;
Ast = 80;
LP_IIR = dsp.LowpassFilter('SampleRate',Fs,'FilterType','IIR',...
    'DesignForMinimumOrder',false,'FilterOrder',Nf,...
    'PassbandFrequency',fp,'PassbandRipple',Ap,'StopbandAttenuation',Ast);
PAin = step(LP_IIR,PAin);%rx_signal_bb_f = LP_IIR(rx_signal_bb);

%% 
t=(0:numel(PAin)-1)/Fs;
IQData = PAin(1:end-N/10).*exp( 1i * 2 * pi * fc(1) * t(1:end-N/10)' )...
    + PAin(N/50+1:end-N/50*4).*exp( 1i * 2 * pi * fc(2) * t(1:end-N/10)' )...
    + PAin(N/50*2+1:end-N/50*3).*exp( 1i * 2 * pi * fc(3) * t(1:end-N/10)' );%...
%     + PAin(N/50*3+1:end-N/50*2).*exp( 1i * 2 * pi * fc(4) * t(1:end-N/10)' )...
%     + PAin(N/50*4+1:end-N/50).*exp( 1i * 2 * pi * fc(5) * t(1:end-N/10)' );
x=PAin(1:end);
y=IQData;
PSDx=psd(Hs,x,'Fs',Fs,'CenterDC',true);
X=PSDx.Data;
PX = 10*log10(X/max(X));
PSDy=psd(Hs,y,'Fs',Fs,'CenterDC',true);
Y=PSDy.Data;
PY = 10*log10(Y/max(Y));
f=PSDx.Frequencies;
plot(f,PX,f,PY)
PAin=IQData;

%% Filter

BW=BW*numel(fc);
Fs_N=200e6;
PAin=resample(PAin,Fs_N,Fs);
Fs=Fs_N;
Nf   = 10;
fp  = BW*0.5; %*(Kav(k)+1)
Ap  = 0.001;
Ast = 80;
LP_IIR = dsp.LowpassFilter('SampleRate',Fs,'FilterType','IIR',...
    'DesignForMinimumOrder',false,'FilterOrder',Nf,...
    'PassbandFrequency',fp,'PassbandRipple',Ap,'StopbandAttenuation',Ast);
PAin = step(LP_IIR,PAin);%rx_signal_bb_f = LP_IIR(rx_signal_bb);

spectraplot(PAin,Fs)

save(['PAinLTE',num2str(BW*1e-6),'MHz'], 'BW','Fs','PAin')