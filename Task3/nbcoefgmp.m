

function NbCoeff=nbcoefgmp(Model)

if strcmpi(Model.Type,'mp')
    Model.NL(2).Kv=[]; Model.Mem(2).Lv=[]; Model.Mem(2).Mv=[];
    Model.NL(3).Kv=[]; Model.Mem(3).Lv=[]; Model.Mem(3).Mv=[];
end

if isfield(Model,'Symmetric') && strcmpi(Model.Symmetric,'on')
    Model.NL(3)=Model.NL(2);
    Model.Mem(3)=Model.Mem(2);
end

NbCoeff=numel(Model.NL(1).Kv)*numel(Model.Mem(1).Lv)+...
        numel(Model.NL(2).Kv)*numel(Model.Mem(2).Lv)*...
        numel(Model.Mem(2).Mv)+...
        numel(Model.NL(3).Kv)*numel(Model.Mem(3).Lv)*...
        numel(Model.Mem(3).Mv);