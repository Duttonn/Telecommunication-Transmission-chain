function [Model,Eval] = gmpidentifier(Data, Model, Options)
%
% [Model,Eval] = gmpidentifier(Data, Model)
%
% gmpidentifier identifies the generalized memory polynomial model given 
% the input output signals and the appropriate orders. This model is 
% divided into three separate expressions, the first one is the memory 
% polynomial model i.e., terms on the diagonal, the second [third] one is 
% for lagging [leading] envelope cross terms.
% 
% Input Arguments:
% Data  : Structure with two fields 
%       In     : time domain input signal.
%       Out     : time domain output signal.
% Model :  structure with 4 fields:
    % NL    : NonLinearity, 1x3 structure array, 
        % NL(1) is for the MP model and the field inside is Kv the vector of 
        % indices for MP,
        % NL(2) [NL(3)] is for the second [third] nonlinearity and the field inside
        % is Kv vector of indices for lagging [leading] envelope cross terms.
    % Mem   : Memory, 1x3 structure array,
        % Mem(1) is the memory for MP, with one field Lv a vector of indices for
        % memory.
        % Mem(2) [Mem(3)] is the memory for lagging [leading] envelope cross terms, 
        % containing two fields Lv and Mv vectors for the indices
        % of memory.
    % Symmetric : A string, if equal to one the parameters to control the
      % leading terms don't have to be included, they will be chosen equal to
      % those of lagging terms.
% 
% Output arguments:
% Model  : the input argument with one additional field Coeff (that may be 
% already existing in the input argument but will be replaced by this 
% function) vector of all the Coefficients akl, bklm and cklm with the 
% summation on NL (k) preceding the summations on memory (l and m). akl are 
% the Coefficients of the Memory Polynomial (MP) Part and bklm and cklm are
% the Coefficients of lagging and leading envelope terms respectively. 
% Example: 
% For NL(1).Kv=[0 1], Mem(1).Lv=[0 1]
% NL(2).Kv=[1], Mem(2).Lv=[0], Mem(2).Mv=[1 2]
% NL(3).Kv=[], Mem(3).Lv=[], Mem(3).Mv=[]
% => av=[a00 a01 a10 a11 b101 b102]^T.
% 
% MSE    : Mean-Square-Error
% 
% The Input Matrices are included in 3 steps:
%
% step 1: Terms on the diagonal (quasi-memoryless model i.e.
% polynomial model and memory polynomial model with memory), orders are
% defined as vectors of indices, Model.NL(1).Kav and Model.Mem(1).Lav.
% Generated terms are of the form: x(n-l)|x(n-l)|^k, l taking values from
% Lav, and k from Kav.
% Example: NL(1).Kv=[0 2] and Mem(1).Lv=[0 2], terms from step 1 are:
% x(n)+x(n-2)+x(n)|x(n)|^2+x(n-2)|x(n-2)|^2.
%
% step 2: cross terms signal and lagging envelope.
% Orders are defined as vectors of indices, Model.NL(2).Kv, 
% Model.Mem(2).Lv and Model.Mem(2).Mv.
% the first element of Kv and Mv must not be zero in order to avoid
% redundancy.
% Generated terms are of the form: x(n-l)|x(n-l-m)|^k, l taking values from
% Mem(2).Lv, m from Mem(2).Mv and k from NL(2).Kv.
%
% step 3: cross terms signal and leading envelope.
% Orders are defined as vectors of indices, Model.NL(3).Kv, 
% Model.Mem(3).Lv and Model.Mem(3).Mv.
% the first element of Kcv and Mcv must not be zero in order to avoid
% redundancy.
% Generated terms are of the form: x(n-l)|x(n-l+m)|^k, l taking values from
% Mem(3).Lv, m from Mem(3).Mv and k from NL(3).Kv.
%
% (c) Mazen Abi Hussein, 2008
%
if nargin<3
    Options.Evaluation.Enable='off';
end
if ~isfield(Model,'Type')
    Model.Type='mp';
end
if strcmpi(Model.Type,'mp')
    Model.NL(2).Kv=[]; Model.Mem(2).Lv=[]; Model.Mem(2).Mv=[];
    Model.NL(3).Kv=[]; Model.Mem(3).Lv=[]; Model.Mem(3).Mv=[];
end
if ~isfield(Model,'Symmetric')
    Model.Symmetric='off';
end
if strcmpi(Model.Symmetric,'on')
    Model.NL(3)=Model.NL(2);
    Model.Mem(3)=Model.Mem(2);
end

Kav=Model.NL(1).Kv;
Lav=Model.Mem(1).Lv;
Kbv=Model.NL(2).Kv;
Lbv=Model.Mem(2).Lv;
Mbv=Model.Mem(2).Mv;
Kcv=Model.NL(3).Kv;
Lcv=Model.Mem(3).Lv;
Mcv=Model.Mem(3).Mv;

% To include later
% minargs=3;
% maxargs=10;
% 
% narginchk(minargs, maxargs)

% switch nargin
%     case 2
%         error ('At least one order vector must be defined.')
%     case 3
%         Lav=[]; Kbv=[]; Lbv=[]; Mbv=[]; Kcv=[]; Lcv=[]; Mcv=[];
%     case 4
%         Kbv=[]; Lbv=[]; Mbv=[]; Kcv=[]; Lcv=[]; Mcv=[];
%     case 5
%         lastwarn('Memory indices are not defined,',...
%             ' this will lead to redundancy in terms.')
%         Lbv=[]; Mbv=[]; Kcv=[]; Lcv=[]; Mcv=[];
%     case 6
%         lastwarn('Memory indices for lagging envelope are not defined,',...
%             ' this will lead to redundancy in terms.')
%         Mbv=[]; Kcv=[]; Lcv=[]; Mcv=[];
%     case 7
%         Kcv=[]; Lcv=[]; Mcv=[];
%     case 8
%         lastwarn('Memory indices are not defined,',...
%             ' this will lead to redundancy in terms.')
%         Lcv=[]; Mcv=[];
%     case 9
%         lastwarn('Memory indices for leading envelope are not defined,',...
%             ' this will lead to redundancy in terms.')
%         Mcv=[];
% end

x=Data.In;
y=Data.Out;
S=size(x);
if S(1)==1 % line vector
    x=x(:);
end
S=size(y);
if S(1)==1 % line vector
    y=y(:);
end

A1=[];
for k=1:length(Kav)
    for l=1:length(Lav)
        cv=[zeros(Lav(l),1) ; x(1:end-Lav(l))];
        A1= [A1 cv.*(abs(cv).^(Kav(k)))];
    end
end


% cross terms signal and lagging envelope
A2=[];
for k=1:length(Kbv)
    for l=1:length(Lbv)
        cv1=[zeros(Lbv(l),1) ; x(1:end-Lbv(l))];
        for m=1:length(Mbv)
            cv2=[zeros(Lbv(l)+Mbv(m),1); x(1:end-Lbv(l)-Mbv(m))];
            A2=[A2 cv1.*(abs(cv2).^Kbv(k))];
        end
    end
end

% cross terms signal and leading envelope
A3=[];
for k=1:length(Kcv)
    for l=1:length(Lcv)
        cv1=[zeros(Lcv(l),1) ; x(1:end-Lcv(l))];
        for m=1:length(Mcv)
            cv2=[zeros(Lcv(l)-Mcv(m),1); x((Lcv(l)<Mcv(m))*...
                (Mcv(m)-Lcv(l))+1:end-(Lcv(l)>Mcv(m))*...
                (Lcv(l)-Mcv(m)));zeros(Mcv(m)-Lcv(l),1)];
            A3=[A3 cv1.*(abs(cv2).^Kcv(k))];
        end
    end
end

% x_1=[0;x(1:end-1)];
% angv=[angle(x)-angle(x_1)];
% A=exp(1j.*angv).*[A1 A2 A3];

A=[A1 A2 A3];

CondMat=cond(A'*A);
Eval.MatCondNb=CondMat;
% disp(CondMat)

av=(A'*A)\(A'*y);

CondMat=cond(A'*A);
Eval.MatCondNb=CondMat;
EvalData.Est=A*av;
EvalData.Meas=y;
NMSE=nmse(EvalData);
Eval.NMSE=NMSE;%.Z;
% disp(CondMat)

Model.Coeff=av;
