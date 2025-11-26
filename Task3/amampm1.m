function PA=amampm1(PA,DisplayOptions)
%
% PA=amampm(Data,disp)
% This function takes time domain input/output signals of a PA, and gives
% all its dynamic characteristics (relative to the modulation used).
% Input Arguments:
%       PA: is a structure with 2 fields and an additional  optional one.
%           Input   : The input signal.
%           Output  : The output signal.
%           ALimLinIn : The highest small signal amplitude
%                       this variable is optional, and its default value is
%                       equal to max(|In|)/2.
%       DisplayOptions: is a structure with many fields of type string and
%                       that can be 'on' or 'off':
%                           AMAMPM    : to activate desactivate the AM/AM
%                           and AM/PM curves
%                       It's an optional argument, with a default
%       value 'off'.
% To be completed later.
% Output argument:
%       PA: To be completed later
%
% (c) Mazen Abi Hussein, 2012
% updated 140516: replacing Data argument with PA
% updated 150930: finding linear zone of PA


if nargin<2
    DisplayOptions.AMAMPM.LinVolts.Enable='off';
    DisplayOptions.AMAMPM.PwrWatts.Enable='off';
    DisplayOptions.AMAMPM.PwrdBm.Enable='off';
end

if ~isfield(DisplayOptions.AMAMPM,'NbSamp')
    DisplayOptions.AMAMPM.NbSamp=numel(PA.Input.IQdata);
end
if ~isfield(DisplayOptions.AMAMPM,'LinVolts')
    DisplayOptions.AMAMPM.LinVolts.Enable='off';
end
if ~isfield(DisplayOptions.AMAMPM,'PwrWatts')
    DisplayOptions.AMAMPM.PwrWatts.Enable='off';
end
if ~isfield(DisplayOptions.AMAMPM,'PwrdBm')
    DisplayOptions.AMAMPM.PwrdBm.Enable='off';
end

if ~isfield(DisplayOptions.AMAMPM.LinVolts,'Dispersion')
    DisplayOptions.AMAMPM.LinVolts.Dispersion.Enable='off';
end
if ~isfield(DisplayOptions.AMAMPM.PwrWatts,'Dispersion')
    DisplayOptions.AMAMPM.PwrWatts.Dispersion.Enable='off';
end
if ~isfield(DisplayOptions.AMAMPM.PwrdBm,'Dispersion')
    DisplayOptions.AMAMPM.PwrdBm.Dispersion.Enable='off';
end

if ~isfield(DisplayOptions.AMAMPM.LinVolts,'FigureName')
    DisplayOptions.AMAMPM.LinVolts.FigureName='AM/AM & AM/PM (V)';
end
if ~isfield(DisplayOptions.AMAMPM.PwrWatts,'FigureName')
    DisplayOptions.AMAMPM.PwrWatts.FigureName='AM/AM & AM/PM (W)';
end
if ~isfield(DisplayOptions.AMAMPM.PwrdBm,'FigureName')
    DisplayOptions.AMAMPM.PwrdBm.FigureName='AM/AM & AM/PM (dBm)';
end
if ~isfield(PA,'MinCompSatdB')
    PA.MinCompSatdB=1.3;
end

%% Completing info on input signals
if ~isfield(PA.Input,'AvgPwrdBm')
    PA.Input.AvgPwrdBm=pwrdbm(PA.Input.IQdata);
end
if ~isfield(PA.Output,'AvgPwrdBm')
    PA.Output.AvgPwrdBm=pwrdbm(PA.Output.IQdata);
end

if ~isfield(PA.Input,'AvgPwrW')
    PA.Input.AvgPwrW=dbm2w(PA.Input.AvgPwrdBm);
end
if ~isfield(PA.Output,'AvgPwrW')
    PA.Output.AvgPwrW=dbm2w(PA.Output.AvgPwrdBm);
end

if ~isfield(PA.Input,'PAPR')
    PA.Input.PAPR=papr(PA.Input.IQdata);
end
if ~isfield(PA.Output,'PAPR')
    PA.Output.PAPR=papr(PA.Output.IQdata);
end

% dBm & W conversion
if ~isfield(PA.Input,'InstPwrdBm')
    PA.Input.InstPwrdBm=sigpwrdbm(PA.Input.IQdata);
end
if ~isfield(PA.Input,'InstPwrW')
    PA.Input.InstPwrW=dbm2w(PA.Input.InstPwrdBm);
end
if ~isfield(PA.Output,'InstPwrdBm')
    PA.Output.InstPwrdBm=sigpwrdbm(PA.Output.IQdata);
end
if ~isfield(PA.Output,'InstPwrW')
    PA.Output.InstPwrW=dbm2w(PA.Output.InstPwrdBm);
end
% PA.Input.InstPwrdBm=sigpwrdbm(PA.Input.IQdata);
% PA.Input.InstPwrW=dbm2w(PA.Input.InstPwrdBm);
%
% PA.Output.InstPwrdBm=sigpwrdbm(PA.Output.IQdata);
% PA.Output.InstPwrW=dbm2w(PA.Output.InstPwrdBm);

%% Linear Gain & LPA output
if ~isfield(PA,'ALimLinIn')
    PA.ALimLinIn=0.6*max(abs(PA.Input.IQdata));
end
PA.ALimLinIndBm=v2dbm(PA.ALimLinIn);
PA.TotalGain=abs(PA.Output.IQdata)./abs(PA.Input.IQdata);%add by Siqi
% PA.MeanGain=mean(PA.TotalGain);%add by Siqi
PA.StdGain=std(PA.TotalGain);
% Iv=find(abs(PA.TotalGain)>(PA.MeanGain));%add by Siqi
Iv=find(abs(PA.Input.IQdata)<PA.ALimLinIn & ...
    abs(PA.Input.IQdata)>(8/100)*PA.ALimLinIn); % 8% to avoid noise
PA.LinGv=abs(PA.Output.IQdata(Iv))./abs(PA.Input.IQdata(Iv));
PA.LinGdBv=lin2db(PA.LinGv);

if isfield(PA,'Version')
    if strcmpi(PA.Version,'DohertyPAMP')||strcmpi(Data.Version,'DohertyPA')
%         PA.G=mean(abs(PAout))./mean(abs(PAin));
        PA.GdB=pwrdbm(PA.Output.IQdata)-pwrdbm(PA.Input.IQdata);
        PA.G=10^(PA.GdB/20);
    else
        PA.G=mean(PA.LinGv);
    end
else
    PA.G=mean(PA.LinGv);
end
% PA.G=PA.StdGain; %added by Siqi
PA.GdB=lin2db(PA.G);

PA.IvLinGv=Iv;

% Linear PA output
PA.OutputLin=PA.Output;
PA.OutputLin.IQdata=PA.G*PA.Input.IQdata;

PA.OutputLin.InstPwrdBm=sigpwrdbm(PA.OutputLin.IQdata);
PA.OutputLin.InstPwrW=dbm2w(PA.OutputLin.InstPwrdBm);

%% Dispersion in AM/AM
[PA.MaxG,PA.IndexMaxG]=max(PA.LinGv);
PA.MinG=min(PA.LinGv);

PA.stdG=std(PA.LinGv);
PA.stdGdB=lin2db(PA.stdG);

PA.G1stdUpper=PA.G+PA.stdG;         PA.GdB1stdUpper=lin2db(PA.G1stdUpper);
PA.G1stdLower=PA.G-PA.stdG;         PA.GdB1stdLower=lin2db(PA.G1stdLower);
PA.G2stdUpper=PA.G+2*PA.stdG;       PA.GdB2stdUpper=lin2db(PA.G2stdUpper);
PA.G2stdLower=PA.G-2*PA.stdG;       PA.GdB2stdLower=lin2db(PA.G2stdLower);

PA.stdGPerc=PA.stdG*100/PA.G;

u1=abs(PA.Input.IQdata(PA.IvLinGv)*PA.G1stdUpper);
l1=abs(PA.OutputLin.IQdata(PA.IvLinGv));
PA.OffsetDispG1std=max(abs(u1-l1));
PA.OffsetDispW1std=max(abs(v2w(u1)-v2w(l1)));
% PA.OffsetDispW1std=v2w(PA.OffsetDispG1std); %added by Siqi

PA.IvOutliers1std=find(abs(PA.Output.IQdata)>(abs(PA.OutputLin.IQdata)+...
    PA.OffsetDispG1std));
%% Output/Input Saturation and Gain at saturation
[PA.InSatV,I]=max(abs(PA.Input.IQdata));
PA.OutSatV=abs(PA.Output.IQdata(I));  
% [PA.OutSatV,I]=max(abs(PA.Output.IQdata)); 
% PA.InSatV=abs(PA.Input.IQdata(I));
[~,I]=min(abs(abs(PA.OutputLin.IQdata)-PA.OutSatV));
PA.OutLPASatV=abs(PA.OutputLin.IQdata(I));
FlagSat=1;
DiffSat=sigpwrdbm(max(abs(PA.OutputLin.IQdata)))-sigpwrdbm(PA.OutSatV);

if DiffSat<=PA.MinCompSatdB
    PA.OutSatV=NaN;
    PA.InSatV=NaN;
    PA.OutLPASatV=NaN;
    FlagSat=0;
end


PA.OutSatdBm=sigpwrdbm(PA.OutSatV);
PA.OutSatW=dbm2w(PA.OutSatdBm);

PA.InSatdBm=sigpwrdbm(PA.InSatV);
PA.InSatW=dbm2w(PA.InSatdBm);

PA.Gsat=PA.OutSatV/PA.InSatV;
PA.GsatdB=lin2db(PA.Gsat);

% Linear PA: Output/Input Saturation

PA.OutLPASatdBm=sigpwrdbm(PA.OutLPASatV);
PA.OutLPASatW=dbm2w(PA.OutLPASatdBm);

PA.InLPASatV=PA.OutLPASatV/PA.G;
PA.InLPASatdBm=sigpwrdbm(PA.InLPASatV);
PA.InLPASatW=dbm2w(PA.InLPASatdBm);

PA.CompAtSatdB=PA.GdB-PA.GsatdB;
%% 1dB compression
Iv=PA.Input.InstPwrdBm>PA.ALimLinIndBm;
PA.CompRegionIn=PA.Input.InstPwrdBm(Iv);
PA.CompRegionOut=PA.Output.InstPwrdBm(Iv);
PA.CompRegionLinOut=PA.OutputLin.InstPwrdBm(Iv);
Diff=PA.CompRegionLinOut-PA.CompRegionOut;
% [Mini,I]=min(Diff-1);
M=1e-2;
Iv=Diff<=1+M & Diff>=1-M;
% if ~isempty(Iv) % no need because if it's empty the plot function won't
% generate a plot
PA.IP1dBdBm=mean(PA.CompRegionIn(Iv));
PA.OP1dBdBm=mean(PA.CompRegionOut(Iv));
PA.IP1dBW=dbm2w(PA.IP1dBdBm);
PA.OP1dBW=dbm2w(PA.OP1dBdBm);
PA.IP1dBV=dbm2v(PA.IP1dBdBm);
PA.OP1dBV=dbm2v(PA.OP1dBdBm);
%     FlagComp1dB=1;
% end
%% Info for the normalization of the PD Input
PA.LimitPDin=PA.InLPASatV;
PA.LimitPDinNorm=PA.LimitPDin/PA.InSatV;
PA.LimitPDindBm=PA.InLPASatdBm;
PA.LimitPDinW=dbm2w(PA.LimitPDindBm);

PA.PDIBOdB=PA.InSatdBm-PA.InLPASatdBm;


%% Phase shift
phi_in=unwrap(angle(PA.Input.IQdata));
phi_out=unwrap(angle(PA.Output.IQdata));

PhaseShift = mod(phi_out-phi_in,2*pi);%unwrap(phi_out-phi_in); mod(phi_out-phi_in,2*pi)
Iv=find(PhaseShift>pi);
PhaseShift(Iv)=PhaseShift(Iv)-2*pi;
% Iv=find(PhaseShift<-pi);
% PhaseShift(Iv)=PhaseShift(Iv)+2*pi;
% PhaseShift=PhaseShift-pi;
PhaseShift=rad2deg(PhaseShift);

PA.PhaseShift=PhaseShift;
PA=orderfields(PA);
%% Figure LinVolts
NbSmp=DisplayOptions.AMAMPM.NbSamp;
if strcmpi(DisplayOptions.AMAMPM.LinVolts.Enable,'on')
    UpperLimitYaxis=1.1*max(abs(PA.Output.IQdata));
    RangeYaxis=UpperLimitYaxis;
    LowerLimitYaxis=UpperLimitYaxis-RangeYaxis;
    UpperLimitXaxis=1.1*max(abs(PA.Input.IQdata));
    RangeXaxis=UpperLimitXaxis;
    LowerLimitXaxis=UpperLimitXaxis-RangeXaxis;
    
    h=figure('WindowStyle','docked','NumberTitle','off');
    set(h,'Name',DisplayOptions.AMAMPM.LinVolts.FigureName)
    [AX,H1,H2]=plotyy(abs(PA.Input.IQdata(1:NbSmp)), abs(PA.Output.IQdata(1:NbSmp)),...
        abs(PA.Input.IQdata(1:NbSmp)), PA.PhaseShift(1:NbSmp));
    set([H1,H2],'Marker','.','LineStyle','no')%,'LineStyle','.'
    set(H1,'DisplayName','AM/AM')
    set(H2,'DisplayName','AM/PM')
    hold on
    plot(AX(1),abs(PA.Input.IQdata(1:NbSmp)),abs(PA.OutputLin.IQdata(1:NbSmp)),...
        'k','Marker','.','LineStyle','no','DisplayName','LPA');
    if FlagSat
        plot(AX(1),[0 PA.InSatV],[PA.OutSatV PA.OutSatV],'r',...
            'LineStyle','--',...
            'DisplayName',['OutSatV=',num2str(PA.OutSatV,3)]);
        plot(AX(1),[PA.InLPASatV PA.InLPASatV],[0 PA.OutLPASatV],...
            'r','LineStyle','-.',...
            'DisplayName',['LPAInSatV=',num2str(PA.InLPASatV,3)]);
        plot(AX(1),PA.InLPASatV,PA.OutLPASatV,'LineStyle','no',...
            'LineWidth',2,'MarkerEdgeColor','r','MarkerSize',8,...
            'DisplayName','Saturation LPA','Marker','o');
    end
    
    if strcmpi(DisplayOptions.AMAMPM.LinVolts.Dispersion.Enable,'on')
        plot(AX(1),abs(PA.Input.IQdata(1:NbSmp)),...
            abs(PA.OutputLin.IQdata(1:NbSmp))+PA.OffsetDispG1std,'r','Marker','.',...
            'LineStyle','no','DisplayName','UpperLimitDispersion1std',...
            'MarkerSize',4);
        plot(AX(1),abs(PA.Input.IQdata(1:NbSmp)),...
            abs(PA.OutputLin.IQdata(1:NbSmp))-PA.OffsetDispG1std,'r','Marker','.',...
            'LineStyle','no','DisplayName','LowerLimitDispersion1std',...
            'MarkerSize',4);
    end
    plot(AX(1),PA.IP1dBV,PA.OP1dBV,...
        'r','Marker','*','DisplayName',...
        ['1dB: IP1dBV=',num2str(PA.IP1dBV,3),...
        ', OP1dBV=',num2str(PA.OP1dBV,3)],'MarkerSize',8);%,...
        % 'MarkerEdgeColor','k');
    % Avg
    AvgIn=mean(abs(PA.Input.IQdata(1:NbSmp)));
    AvgOut=mean(abs(PA.Output.IQdata(1:NbSmp)));
    plot(AX(1),[LowerLimitXaxis AvgIn],...
        [AvgOut AvgOut],'r',...
        'LineStyle','--',...
        'DisplayName',['OutAvg=',num2str(AvgOut,3)]);
    plot(AX(1),[AvgIn AvgIn],...
        [LowerLimitYaxis AvgOut],...
        'r','LineStyle','-.',...
        'DisplayName',['InAvg=',num2str(AvgIn,3)]);
    plot(AX(1),AvgIn,AvgOut,...
        'LineStyle','no',...
        'LineWidth',2,'MarkerEdgeColor','r','MarkerSize',8,...
        'DisplayName','Avg (V)','Marker','s');
    hold off
    xlabel('|PAin|')
    set(get(AX(1),'YLabel'),'String','|PAout|')
    set(get(AX(2),'YLabel'),'String', 'Phase Shift (°)')
    legend('hide')
    grid on
    Pv=PA.PhaseShift(abs(PA.Input.IQdata(1:NbSmp))>mean(abs(PA.Input.IQdata(1:NbSmp))));
    M=4*round(max(abs(Pv)));
    if M==0
        M=2;
        set(AX(2),'YLim',[-M M],'YTick',-M:2*M/10:M)
    else
        set(AX(2),'YLim',[-M M],'YTick',-M:2*M/10:M)
    end
    
    
    vectTicks=LowerLimitYaxis:RangeYaxis/10:UpperLimitYaxis;
    StrVectT=num2str(vectTicks,2);
    vectTicks=str2num(StrVectT); %#ok<*ST2NM>
    set(AX(1),'YLim',[LowerLimitYaxis UpperLimitYaxis],'YTick',vectTicks)
    
    
    vectTicks=LowerLimitXaxis:RangeXaxis/10:UpperLimitXaxis;
    StrVectT=num2str(vectTicks,2);
    vectTicks=str2num(StrVectT); %#ok<*ST2NM>
    set([AX(1),AX(2)],'XLim',[LowerLimitXaxis UpperLimitXaxis],...
        'XTick',vectTicks)
end

%% Figure PwrdBm
if strcmpi(DisplayOptions.AMAMPM.PwrdBm.Enable,'on')
    UpperLimitYaxis=1.1*max(PA.Output.InstPwrdBm);
    if ~isfield(DisplayOptions.AMAMPM.PwrdBm,'RangeYaxis')
        RangeYaxis=20;
    else
        RangeYaxis=DisplayOptions.AMAMPM.PwrdBm.RangeYaxis;
    end
    LowerLimitYaxis=UpperLimitYaxis-RangeYaxis;
    UpperLimitXaxis=max(PA.Input.InstPwrdBm)+2;
    if ~isfield(DisplayOptions.AMAMPM.PwrdBm,'RangeXaxis')
        RangeXaxis=20;
    else
        RangeXaxis=DisplayOptions.AMAMPM.PwrdBm.RangeXaxis;
    end
    LowerLimitXaxis=UpperLimitXaxis-RangeXaxis;
    
    h=figure('WindowStyle','docked','NumberTitle','off');
    set(h,'Name',DisplayOptions.AMAMPM.PwrdBm.FigureName)
    [AX,H1,H2]=plotyy(PA.Input.InstPwrdBm(1:NbSmp), PA.Output.InstPwrdBm(1:NbSmp),...
        PA.Input.InstPwrdBm(1:NbSmp), PA.PhaseShift(1:NbSmp));
    set([H1,H2],'Marker','.','LineStyle','no')%,'LineStyle','.'
    set(H1,'DisplayName','AM/AM')
    set(H2,'DisplayName','AM/PM')
    hold on
    plot(AX(1),PA.Input.InstPwrdBm(1:NbSmp),PA.OutputLin.InstPwrdBm(1:NbSmp),...
        'k','Marker','.','LineStyle','no','DisplayName','LPA');
    if FlagSat
        plot(AX(1),[LowerLimitXaxis PA.InSatdBm],...
            [PA.OutSatdBm PA.OutSatdBm],'r',...
            'LineStyle','--',...
            'DisplayName',['OutSatPwrdBm=',num2str(PA.OutSatdBm,3)]);
        plot(AX(1),[PA.InLPASatdBm PA.InLPASatdBm],...
            [LowerLimitYaxis PA.OutLPASatdBm],...
            'r','LineStyle','-.',...
            'DisplayName',['LPAInSatPwrdBm=',num2str(PA.InLPASatdBm,3)]);
        plot(AX(1),PA.InLPASatdBm,PA.OutLPASatdBm,'LineStyle','no',...
            'LineWidth',2,'MarkerEdgeColor','r','MarkerSize',8,...
            'DisplayName','Saturation LPA','Marker','o');
    end
    if strcmpi(DisplayOptions.AMAMPM.PwrdBm.Dispersion.Enable,'on')
        plot(AX(1),PA.Input.InstPwrdBm(1:NbSmp),...
            PA.Input.InstPwrdBm(1:NbSmp)+PA.GdB1stdUpper,'r','Marker','.',...
            'LineStyle','no','DisplayName','UpperLimitDispersion1std',...
            'MarkerSize',4);
        plot(AX(1),PA.Input.InstPwrdBm(1:NbSmp),...
            PA.Input.InstPwrdBm(1:NbSmp)+PA.GdB1stdLower,'r','Marker','.',...
            'LineStyle','no','DisplayName','LowerLimitDispersion1std',...
            'MarkerSize',4);
        %     if FlagComp1dB==1
        plot(AX(1),PA.IP1dBdBm,PA.OP1dBdBm,...
            'r','Marker','*','DisplayName',...
            ['1dB: IP1dB=',num2str(PA.IP1dBdBm,3),...
            ', OP1dB=',num2str(PA.OP1dBdBm,3)],'MarkerSize',6);
        %     end
    end
    % AvgPwr
    plot(AX(1),[LowerLimitXaxis PA.Input.AvgPwrdBm],...
        [PA.Output.AvgPwrdBm PA.Output.AvgPwrdBm],'r',...
        'LineStyle','--',...
        'DisplayName',['OutAvgPwrdBm=',num2str(PA.Output.AvgPwrdBm,3)]);
    plot(AX(1),[PA.Input.AvgPwrdBm PA.Input.AvgPwrdBm],...
        [LowerLimitYaxis PA.Output.AvgPwrdBm],...
        'r','LineStyle','-.',...
        'DisplayName',['InAvgPwrdBm=',num2str(PA.Input.AvgPwrdBm,3)]);
    plot(AX(1),PA.Input.AvgPwrdBm,PA.Output.AvgPwrdBm,...
        'LineStyle','no',...
        'LineWidth',2,'MarkerEdgeColor','r','MarkerSize',8,...
        'DisplayName','Avg Pwr (dBm)','Marker','s');
    hold off
    xlabel('Instantenous Input Power (dBm)')
    set(get(AX(1),'YLabel'),'String','Instantenous Output Power (dBm)')
    set(get(AX(2),'YLabel'),'String', 'Phase Shift (°)')
    legend('hide')
    grid on
    Pv=PA.PhaseShift(abs(PA.Input.IQdata)>mean(abs(PA.Input.IQdata)));
    M=4*round(max(abs(Pv)));
    if M==0
        M=2;
        set(AX(2),'YLim',[-M M],'YTick',-M:2*M/10:M)
    else
        set(AX(2),'YLim',[-M M],'YTick',-M:2*M/10:M)
    end
    
    vectTicks=LowerLimitYaxis:RangeYaxis/10:UpperLimitYaxis;
    StrVectT=num2str(vectTicks,2);
    vectTicks=str2num(StrVectT); %#ok<*ST2NM>
    set(AX(1),'YLim',[LowerLimitYaxis UpperLimitYaxis],'YTick',vectTicks)
    
    vectTicks=LowerLimitXaxis:RangeXaxis/10:UpperLimitXaxis;
    StrVectT=num2str(vectTicks,2);
    vectTicks=str2num(StrVectT); %#ok<*ST2NM>
    set([AX(1),AX(2)],'XLim',[LowerLimitXaxis UpperLimitXaxis],...
        'XTick',vectTicks)
    
    %     set([AX(1),AX(2)],'YLimMode','auto')
    %     set([AX(1),AX(2)],'XLimMode','auto')
end

%% Figure PwrW
if strcmpi(DisplayOptions.AMAMPM.PwrWatts.Enable,'on')
    UpperLimitYaxis=1.1*max(PA.Output.InstPwrW);
    RangeYaxis=UpperLimitYaxis;
    LowerLimitYaxis=UpperLimitYaxis-RangeYaxis;
    UpperLimitXaxis=1.1*max(PA.Input.InstPwrW);
    RangeXaxis=UpperLimitXaxis;
    LowerLimitXaxis=UpperLimitXaxis-RangeXaxis;
    
    h=figure('WindowStyle','docked','NumberTitle','off');
    set(h,'Name',DisplayOptions.AMAMPM.PwrWatts.FigureName)
    [AX,H1,H2]=plotyy(PA.Input.InstPwrW(1:NbSmp), PA.Output.InstPwrW(1:NbSmp),...
        PA.Input.InstPwrW(1:NbSmp), PA.PhaseShift(1:NbSmp));
    set([H1,H2],'Marker','.','LineStyle','no')%,'LineStyle','.'
    set(H1,'DisplayName','AM/AM')
    set(H2,'DisplayName','AM/PM')
    hold on
    plot(AX(1),PA.Input.InstPwrW(1:NbSmp),PA.OutputLin.InstPwrW(1:NbSmp),...
        'k','Marker','.','LineStyle','no','DisplayName','LPA');
    if FlagSat
        plot(AX(1),[LowerLimitXaxis PA.InSatW],...
            [PA.OutSatW PA.OutSatW],'r',...
            'LineStyle','--',...
            'DisplayName',['OutSatPwrW=',num2str(PA.OutSatW,3)]);
        plot(AX(1),[PA.InLPASatW PA.InLPASatW],...
            [LowerLimitYaxis PA.OutLPASatW],...
            'r','LineStyle','-.',...
            'DisplayName',['LPAInSatPwrW=',num2str(PA.InLPASatW,3)]);
        plot(AX(1),PA.InLPASatW(1:NbSmp),PA.OutLPASatW(1:NbSmp),'LineStyle','no',...
            'LineWidth',2,'MarkerEdgeColor','r','MarkerSize',8,...
            'DisplayName','Saturation LPA','Marker','o');
    end
    if strcmpi(DisplayOptions.AMAMPM.PwrWatts.Dispersion.Enable,'on')
        plot(AX(1),PA.Input.InstPwrW,...
            PA.OutputLin.InstPwrW+PA.OffsetDispW1std,'r','Marker','.',...
            'LineStyle','no','DisplayName','UpperLimitDispersion1std',...
            'MarkerSize',4);
        plot(AX(1),PA.Input.InstPwrW,...
            PA.OutputLin.InstPwrW-PA.OffsetDispW1std,'r','Marker','.',...
            'LineStyle','no','DisplayName','LowerLimitDispersion1std',...
            'MarkerSize',4);
        %     if FlagComp1dB==1
        plot(AX(1),PA.IP1dBW,PA.OP1dBW,...
            'r','Marker','*','DisplayName',...
            ['1dB: IP1dBW=',num2str(PA.IP1dBW,3),...
            ', OP1dBW=',num2str(PA.OP1dBW,3)],'MarkerSize',8);
        %     end
    end
    % AvgPwr
    plot(AX(1),[LowerLimitXaxis PA.Input.AvgPwrW],...
        [PA.Output.AvgPwrW PA.Output.AvgPwrW],'r',...
        'LineStyle','--',...
        'DisplayName',['OutAvgPwrW=',num2str(PA.Output.AvgPwrW,3)]);
    plot(AX(1),[PA.Input.AvgPwrW PA.Input.AvgPwrW],...
        [LowerLimitYaxis PA.Output.AvgPwrW],...
        'r','LineStyle','-.',...
        'DisplayName',['InAvgPwrW=',num2str(PA.Input.AvgPwrW,3)]);
    plot(AX(1),PA.Input.AvgPwrW,PA.Output.AvgPwrW,...
        'LineStyle','no',...
        'LineWidth',2,'MarkerEdgeColor','r','MarkerSize',8,...
        'DisplayName','Avg Pwr (W)','Marker','s');
    hold off
    xlabel('Instantenous Input Power (Watts)')
    set(get(AX(1),'YLabel'),'String','Instantenous Output Power (Watts)')
    set(get(AX(2),'YLabel'),'String', 'Phase Shift (°)')
    legend('hide')
    grid on
    Pv=PA.PhaseShift(abs(PA.Input.IQdata(1:NbSmp))>mean(abs(PA.Input.IQdata(1:NbSmp))));
    M=4*round(max(abs(Pv)));
    if M==0
        M=2;
        set(AX(2),'YLim',[-M M],'YTick',-M:2*M/10:M)
    else
        set(AX(2),'YLim',[-M M],'YTick',-M:2*M/10:M)
    end
    
    vectTicks=LowerLimitYaxis:RangeYaxis/10:UpperLimitYaxis;
    StrVectT=num2str(vectTicks,2);
    vectTicks=str2num(StrVectT); %#ok<*ST2NM>
    set(AX(1),'YLim',[LowerLimitYaxis UpperLimitYaxis],'YTick',vectTicks)
    
    
    vectTicks=LowerLimitXaxis:RangeXaxis/10:UpperLimitXaxis;
    StrVectT=num2str(vectTicks,2);
    vectTicks=str2num(StrVectT); %#ok<*ST2NM>
    set([AX(1),AX(2)],'XLim',[LowerLimitXaxis UpperLimitXaxis],...
        'XTick',vectTicks)
    
    %     set([AX(1),AX(2)],'YLimMode','auto')
    %     set([AX(1),AX(2)],'XLimMode','auto')
end