function PA=amampm(Data,disp)
%
% PA=amampm(Data,disp)
% This function takes time domain input/output signals of a PA, and gives
% all its dynamic characteristics (relative to the modulation used).
% Input Arguments:
%       Data: is a structure with 2 fields and an additional  optional one.
%           In : The input signal, Data.In.
%           Out : The output signal, Data.Out.
%           ALimLinIn : The highest small signal amplitude, Data.ALimLinIn,
%                       this variable is optional, and its default value is 
%                       equal to max(|In|)/2.
%       disp: is a string, that can be 'on' or 'off', to display or not the
%       AM/AM and AM/PM curves. It's an optional argument, with a default
%       value 'off'.
% Output argument:
%       PA: To be completed later
%
% (c) Mazen Abi Hussein, 2012 


PAin=Data.In;
PAout=Data.Out;

PA.AvgPwrIn=pwrdbm(PAin);
PA.AvgPwrOut=pwrdbm(PAout);
%% Linear Gain
if isfield(Data,'ALimLinIn')
    PA.ALimLinIn=Data.ALimLinIn;
else
    PA.ALimLinIn=max(abs(PAin))/3;
end
Iv=find((abs(PAin)<PA.ALimLinIn)&(abs(PAin)>max(abs(PAin))/4));
if isfield(Data,'Version')
    if strcmpi(Data.Version,'DohertyPAMP')||strcmpi(Data.Version,'DohertyPA')
%         PA.G=mean(abs(PAout))./mean(abs(PAin));
        PA.GdB=pwrdbm(PAout)-pwrdbm(PAin);
        PA.G=10^(PA.GdB/20);
    else
        PA.G=mean(abs(PAout(Iv))./abs(PAin(Iv)));
    end
else
    PA.G=mean(abs(PAout(Iv))./abs(PAin(Iv)));
end

PA.GdB=20*log10(PA.G);


%% Output/Input Saturation and Gain at saturation
[PA.OutSat,I]=max(abs(PAout));
PA.OutSatdBm=10*log10(PA.OutSat.^2/(50*1e-3));
PA.InSat=abs(PAin(I));
PA.InSatdBm=10*log10(PA.InSat.^2/(50*1e-3));
PA.Gsat=PA.OutSat/PA.InSat;

%% Linear PA: Output/Input Saturation
LPAout=PA.G*PAin;
[~,I]=min(abs(abs(LPAout)-PA.OutSat));
PA.OutLPASat=abs(LPAout(I));
PA.InLPASat=PA.OutLPASat/PA.G;
PA.AttInPD=PA.InLPASat/max(abs(PAin));

%% Info for the normalization of the PD Input
PA.LimitPD=PA.InLPASat;


if nargin<2
    disp='off';
end

if strcmpi(disp,'on')
    phi_in=unwrap(angle(PAin));
    phi_out=unwrap(angle(PAout));
    
    PhaseShift = mod(phi_out-phi_in,2*pi);%unwrap(phi_out-phi_in);
    Iv=find(PhaseShift>pi);
    PhaseShift(Iv)=PhaseShift(Iv)-2*pi;
    
    PhaseShift=rad2deg(PhaseShift);
    
    
    
    h=figure('WindowStyle','docked','NumberTitle','off');
    set(h,'Name','AM/AM & AM/PM')
    [AX,H1,H2]=plotyy(abs(PAin), abs(PAout), abs(PAin), PhaseShift);
    set([H1,H2],'Marker','.','LineStyle','no')%,'LineStyle','.'
    hold on
    p1=plot(AX(1),abs(PAin),abs(LPAout),'k','Marker','.','LineStyle','no');
    p2=plot([0 PA.InSat],[PA.OutSat PA.OutSat],'r','LineStyle','--');
    p3=plot([PA.InLPASat PA.InLPASat],[0 PA.OutLPASat],...
        'r','LineStyle','--');
    p4=plot(PA.InLPASat,PA.OutLPASat,'LineStyle','no','Marker','o',...
        'LineWidth',2,'MarkerEdgeColor','r','MarkerSize',8);
    hold off
    xlabel('|PAin|')
    set(get(AX(1),'YLabel'),'String','|PAout|')
    set(get(AX(2),'YLabel'),'String', 'Phase Shift (?)')
%     legend([H1 H2 p4],{'AM/AM','AM/PM','Saturation LPA'},'NorthWest')
    grid on
    Pv=PhaseShift(abs(PAin)>mean(abs(PAin)));
    M=2*round(max(abs(Pv)));
    set(AX(2),'YLim',[-M M],'YTick',-M:2*M/10:M)
    M=1.1*PA.OutSat;
    vectTicks=0:M/10:M;
    StrVectT=num2str(vectTicks,2);
    vectTicks=str2num(StrVectT);
    set(AX(1),'YLim',[0 M],'YTick',vectTicks)
    set([AX(1),AX(2)],'XLimMode','auto')
end