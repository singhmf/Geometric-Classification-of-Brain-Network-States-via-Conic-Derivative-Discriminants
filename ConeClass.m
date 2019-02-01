function [Out] = ConeClass(Classes,varargin)
%% the machine-learning toolbox


%% Performs leave-one-out cross-validation using the conic classifier.
%% Data is in format [channel x time] for the derivative time series


%% Input should be a cell of cell arrays{class1{ex1., ex2.}..,class2{ex1,.ex2...}...}
%% Optional input 1: Rank(y/n): saves the time series for each under AllCompare.Rank
%% Optional Input 2: NormEach(y/n): whether to normalize each example's covariance matrix
%% Optional Input 3: Thresh2: Eigenvalue threshold for pseudo-inverse and computing geomean of eigenvalues

%% Outputs: MeanCell gives the covariance matrix for each case of leave-one-out per class
%% The second row of AllCompare.Sum gives n-way classification accuracies for each class
%% The second row of PairCompare.Sum{i} gives pairwise classification accuracies for class{i} vs. all other classes (columns)

%% Summary reports have 2 rows for 2 different trial-based assignment rules
%% FIRST row: Assign classes based upon number of timepoints assigned to each class
%% SECOND row: Assign clusters based upon mean confidence across time points (weighted)
%% Pairwise Accuracy: (nClass-1) cells
%% Each cell contains the summary accuracy of comparing class # (Cell number) to all other classes (ordered in columns)
%% Rows are as before;
if isempty(varargin)
    Rank='n';
    NormEach='y';
    Thresh2=10000;
else
    Rank=varargin{1};
    if numel(varargin)>=2
        NormEach=varargin{2};
        if numel(varargin)>=3
            Thresh2=varargin{3};
        else
            Thresh2=10000;
        end
    else 
        NormEach='y';
        Thresh2=10000;
    end
end

if (~strcmpi(Rank(1),'y'))&&(~strcmpi(Rank(1),'n'))
    error('Rank should be y/n')
end




nClass=numel(Classes);
CovCell=cell(1,nClass);
mMean=cell(1,nClass);
WithinCell=cell(1,nClass);
MExcludeCell=cell(1,nClass);
Clsz=zeros(1,nClass);
CovExclude=cell(1,nClass);

if strcmpi(Rank(1),'y')
    for cl=1:nClass
        Clsz(cl)=numel(Classes{cl});
        AllCompare.Rank{cl}=cell(1,nClass);
    end
end

for cl=1:nClass
Clsz(cl)=numel(Classes{cl});
for samp=1:Clsz(cl)
CovCell{cl}(:,:,samp)=cov(Classes{cl}{samp}');
if strcmpi(NormEach(1),'y')
    [~,d]=eig(CovCell{cl}(:,:,samp));
d=abs(d);
dmax=max(d);

if Thresh2~=0
CovCell{cl}(:,:,samp)=CovCell{cl}(:,:,samp)/CGeomean(d(d>=(dmax/Thresh2)));
else
CovCell{cl}(:,:,samp)=CovCell{cl}(:,:,samp)/CGeomean(d(d>0));
end
end

AllCompare.pW{cl}=zeros(Clsz(cl),nClass);
AllCompare.W{cl}=zeros(Clsz(cl),nClass);
AllCompare.NumAssign{cl}=zeros(Clsz(cl),nClass);
PairCompare.mAcc{cl}=zeros(nClass,Clsz(cl));
PairCompare.mW{cl}=zeros(nClass,Clsz(cl));

end

%% Calculate Leave-One-Out Covariances
CovExclude{cl}=DiffMean(CovCell{cl},3);
for samp=1:Clsz(cl)
    
    [u1,d1]=eig(squeeze(CovExclude{cl}(:,:,samp)));
    d1=diag(abs(d1));
    
    d1(d1<(max(d1)/Thresh2))=0;
    d1(d1~=0)=d1(d1~=0).^(-1);
    
    d=d1;
    d1=diag(d1);
   dmax=max(d); 

if Thresh2~=0
 MExcludeCell{cl}(:,:,samp)=sqrt(abs(d1)/CGeomean(d(d>=(dmax/Thresh2))))*u1';
else
 MExcludeCell{cl}(:,:,samp)=sqrt(abs(d1)/CGeomean(d(d>0)))*u1';
end
 WithinCell{cl}{samp}=sum((MExcludeCell{cl}(:,:,samp)*Classes{cl}{samp}).^2,1);
end
[u1,d1]=eig((mean(CovCell{cl},3)));
d1=diag(abs(d1));
   d1(d1<(max(d1)/Thresh2))=0;
    d1(d1~=0)=d1(d1~=0).^(-1);
    d=d1;
dmax=max(d);
d1=diag(d1);

if Thresh2~=0
    mMean{cl}=sqrt(abs(d1)/CGeomean(d(d>=(dmax/Thresh2))))*u1';
else
    mMean{cl}=sqrt(abs(d1)/CGeomean(d(d>0)))*u1';
end
end

MeanCell=cell(1,nClass);
%Pair Compare is in the order {true,false}
for cl=1:nClass
    for samp=1:Clsz(cl)
    for cl2=setdiff(1:nClass,cl)
        MeanCell{cl}{cl2,samp}=sum((mMean{cl2}*Classes{cl}{samp}).^2,1);
        ktemp=MeanCell{cl}{cl2,samp}-WithinCell{cl}{samp};
        PairCompare.mAcc{cl}(cl2,samp)=mean(ktemp>0);
        PairCompare.mW{cl}(cl2,samp)=sum(ktemp)/sum(abs(ktemp));
    end
    clear mTemp
        nTemp=numel(WithinCell{cl}{samp});
        mTemp(cl,:)=WithinCell{cl}{samp}; %#ok<AGROW>
        mTemp(setdiff(1:nClass,cl),:)=reshape([MeanCell{cl}{:,samp}],nTemp,nClass-1)'; %#ok<AGROW>
        [~,IndTemp]=min(mTemp,[],1);
        if strcmpi(Rank(1),'y')
        [~,rTemp]=sort(mTemp);
        [~,Out.Rank{cl}{samp}]=sort(rTemp);
        end
        AllCompare.W{cl}(samp,:)=sum(mTemp,2);
        
        qTemp=repmat(max(mTemp,[],1),nClass,1)-mTemp;
        AllCompare.pW{cl}(samp,:)=mean(qTemp./repmat(sum(qTemp,1),nClass,1),2);
        
        for z=1:nClass
            AllCompare.NumAssign{cl}(samp,z)=mean(IndTemp==z);
        end
        
    end
    
    PairCompare.Sum{cl}=[mean(PairCompare.mAcc{cl}(setdiff(1:nClass,cl),:)>.5,2) mean(PairCompare.mW{cl}(setdiff(1:nClass,cl),:)>0,2)]';
    
    [~,y]=find(AllCompare.NumAssign{cl}==max(AllCompare.NumAssign{cl},[],2));
    [~,x]=find(AllCompare.W{cl}==min(AllCompare.W{cl},[],2));
    AllCompare.Sum(:,cl)=[mean(y==cl);mean(x==cl)];
    
        
end
Out.Cov_All=CovCell;
%% Old Diagnostics Omitted
%Out.AllCompare=AllCompare;
%Out.PairCompare=PairCompare;
%Out.MExcludeCell=MExcludeCell;
%Out.WithinCell=WithinCell;
%Out.MeanCell=MeanCell;
%Out.mMean=mMean;

%% Make Pairwise accuracy matrix
PWmat=zeros(nClass);PCountMat=zeros(nClass);
for iC=1:nClass
tmpVec=setdiff(1:nClass,iC);
PWmat(iC,tmpVec)=PairCompare.Sum{iC}(2,:);
PCountMat(iC,tmpVec)=PairCompare.Sum{iC}(1,:);
end
Out.Weight_Pair=PWmat;
Out.Count_Pair=PCountMat;
%Out.Pairwise_Accu=PairCompare.Sum;
Out.Weight_All=AllCompare.Sum(2,:);
Out.Count_All=AllCompare.Sum(1,:);
Out.PairGuide='The M(i,j) entry for pairs is accuracy for i in the contrast i vs j';

function[Out2]=DiffMean(X2,iDim)
%% Calculates the leave-1-out mean along a specified direction;
%Out=zeros(size(X));
n=size(X2,iDim);
Out2=(sum(X2,iDim)-X2)/(n-1);
end

function[Out3]=CGeomean(V)
%% Efficiently calculates geometric means (similar to geomean in the Machine Learning Toolbox)
    Out3=exp(mean(log(V)));
end

end