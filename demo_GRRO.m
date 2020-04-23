% This is an example file on how the GRRO program [1] could be used.

% J. Zhang, Y. Lin, M. Jiang, S. Li, Y. Tang, K. C. Tan:
% Multi-label feature selection via global relevance and redundancy optimization. In *IJCAI*, Yokohama, Japan, 2020.

% Please feel free to contact me (zhangjia_gl@163.com), if you have any problem about this program.
clear;clc; addpath(genpath('.\'))

load Bibtex_data

para.alpha=100; para.beta=0.1;

numB=2; [train_data train_target]=trans(train_data,train_target,numB); % ÀëÉ¢»¯

[~,num_feature] = size(train_data); [~,num_label] = size(train_target);

Aeq=ones(1,num_feature); FF=zeros(num_feature,num_feature); FL=zeros(num_feature,num_label);

t0 = clock;
for i=1:num_feature
    for j=1:num_label
		% The relevance betwwen feature and label
		FL(i,j)=mi(train_data(:,i),train_target(:,j));
    end 
end

for i=1:num_feature
    for j=1:num_feature
        % The redundancy betwwen features
        FF(i,j)=mi(train_data(:,i),train_data(:,j));
    end
end
for i=1:num_feature
    FF(i,i)=1;
end

% The relevance betwwen labels
LL = pdist2( train_target', train_target', 'cosine' );
ind=find(isnan(LL));LL(ind)=0;

A = eye(num_feature)+para.alpha*FF; B = para.beta*LL; C = -FL;
W = lyap(A, B, C);  
[dumb feature_idx] = sort(sum(W,2),'descend');
time = etime(clock, t0);

load('Bibtex_data.mat')
% The default setting of MLKNN
Num = 10;Smooth = 1;

% Train and test
% If you use MLKNN as the classifier, please cite the literature [2]
% [2] M.-L. Zhang, Z.-H. Zhou:
% ML-KNN: A lazy learning approach to multi-label learning. Pattern Recognition 40(7): 2038-2048 (2007)
for i = 1:50
    fprintf('Running the program with the selected features - %d/%d \n',i,num_feature);
    
    f=feature_idx(1:i);
    [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data(:,f),train_target,Num,Smooth);
    [HammingLoss,RankingLoss,Coverage,Average_Precision,macrof1,microf1,Outputs,Pre_Labels]=...
        MLKNN_test(train_data(:,f),train_target,test_data(:,f),test_target,Num,Prior,PriorN,Cond,CondN);
    
    HL_GRRO(i)=HammingLoss;
    RL_GRRO(i)=RankingLoss;
    CV_GRRO(i)=Coverage;
    AP_GRRO(i)=Average_Precision;
    MA_GRRO(i)=macrof1;
    MI_GRRO(i)=microf1;
end

save('Bibtex_GRRO.mat','HL_GRRO','RL_GRRO','CV_GRRO'...
    ,'AP_GRRO','MA_GRRO','MI_GRRO','W','feature_idx','time');
