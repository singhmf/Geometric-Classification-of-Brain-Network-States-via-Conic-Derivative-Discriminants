function [Out] = ConeClass_wDeriv(Classes,varargin)
%% This Version Additionally Calculates the Normalized Derivatives
for iClass=1:numel(Classes)
    for jTrial=1:numel(Classes{iClass})
        tmp=convn(Classes{iClass}{jTrial},[1 -1],'valid');
        Classes{iClass}{jTrial}=tmp./sqrt(sum(tmp.^2,1));
    end
end

Out=ConeClass(Classes,varargin{:});
end

