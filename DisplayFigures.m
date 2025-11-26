%% Figure Spectra
vectColor=['r';'c';'k';'m';'g'];
NbIter=numel(SystIter);
if NbIter>numel(vectColor)
    vectColor=repmat(vectColor,round(NbIter/numel(vectColor))+1,1);
end

H=[];
h=figure('DefaultAxesFontSize',20,'WindowStyle','docked','NumberTitle','off');
set(h,'Name','Spectra Comparison')
hold on

for n=1:NbIter
    IterNb=Algorithm.NbIterPerStage;
    h=plot(f*1e-6,SystIter(n).PY,vectColor(n));
    H=[H; h];
    set(H(end),'DisplayName',['It. No. ',num2str(n-1)],...
        'LineWidth',2)
end

hold off
% set(H,'LineWidth',1)
xlabel('Frequency (MHz)')
ylabel('Normalized Magnitude (dB/Hz)')
title('LTE')
grid on
% set([H(4:end)],'Visible','off')
set(gca,'YLim',[min(SystIter(n).PY)-5 max(SystIter(n).PY)+5])
set(gca,'XLim',[-Fs*1e-6/2 Fs*1e-6/2])
legend('show')

%% AM/AM
h=figure('DefaultAxesFontSize',20,'WindowStyle','docked','NumberTitle','off');
set(h,'Name','AM/AM')
hold on
for i=1:NbIter
    n=Algorithm.NbIterPerStage;
    %     if mod(i,n)==1
    h=plot(abs(SystIter(i).u(1:1e3)),...
        abs(SystIter(i).y(1:1e3)),...
        vectColor(i),...
        'LineStyle','no','Marker','.');
    set(h,'DisplayName',['It. No. ',num2str(i-1)],'LineWidth',2)
    %     end
end
legend('show','Location','southeast')
hold off
title('AM/AM curves')
grid on

%% Figure ACPR
h=figure('DefaultAxesFontSize',20,'WindowStyle','docked','NumberTitle','off');
set(h,'Name','ACPR Performance')
ACPR1=zeros(NbIter,1);
ACPR_1=ACPR1;
ACPR2=ACPR1;
ACPR_2=ACPR1;
for n=1:NbIter
    ACPR1(n)=SystIter(n).ACPR.U1;
    ACPR2(n)=SystIter(n).ACPR.U2;
    ACPR_1(n)=SystIter(n).ACPR.L1;
    ACPR_2(n)=SystIter(n).ACPR.L2;
end
plot(0:NbIter-1,ACPR1,...
    0:NbIter-1,ACPR2,...
    0:NbIter-1,ACPR_1,...
    0:NbIter-1,ACPR_2)
legend('ACPR1','ACPR2','ACPR-1','ACPR-2')
xlabel('Iteration No.')
grid on

%% Figure EVM
for i=1:NbIter
    In=SystIter(i).u;
    Out=fixpwr(SystIter(i).y,pwrdbm(In));
    diff=Out-In;
    EVMLin(i)=std(diff)/std(In);
    EVMp(i)=100*EVMLin(i);
end

h=figure('DefaultAxesFontSize',20,'WindowStyle','docked','NumberTitle','off');
set(h,'Name','EVM Performance')
%     semilogy(0:NbIter-1,EVMLin)
plot(0:NbIter-1,EVMp)
xlabel('No. of iterations')
legend('EVM')
ylabel('EVM (%)')
grid on
