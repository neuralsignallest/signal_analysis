% 20170319

close all; clear all; tic;
dsr = 30;
INPUT_DATA = csvread('DOWN_Plot_and_Store_Combined.csv',0,0) ;
down_data = load_data(INPUT_DATA);
INPUT_DATA = csvread('UP_Plot_and_Store_Combined.csv',0,0) ;
up_data = load_data(INPUT_DATA);
INPUT_DATA = csvread('LEFT_Plot_and_Store_Combined.csv',0,0) ;
left_data = load_data(INPUT_DATA);
INPUT_DATA = csvread('RIGHT_Plot_and_Store_Combined.csv',0,0) ;
right_data = load_data(INPUT_DATA);
INPUT_DATA = csvread('PUSH_Plot_and_Store_Combined.csv',0,0) ;
push_data = load_data(INPUT_DATA);
INPUT_DATA = csvread('PULL_Plot_and_Store_Combined.csv',0,0) ;
pull_data = load_data(INPUT_DATA);

labeldown = label5(down_data,1);
labelup = label5(up_data,2);
labelleft = label5(left_data,3);
labelright = label5(right_data,4);
labelpush = label5(push_data,5);
labelpull = label5(pull_data,6);


[labelup,up_data] = labelseg(labelup,up_data, 7);  
[labeldown,down_data] = labelseg(labeldown,down_data, 7); 
[labelleft,left_data] = labelseg(labelleft,left_data, 7); 
[labelright,right_data] = labelseg(labelright,right_data, 7); 
[labelpush,push_data] = labelseg(labelpush,push_data, 7); 
[labelpull,pull_data] = labelseg(labelpull,pull_data, 7); 
% 
labelup = zeroreplace(labelup, 7);
labeldown = zeroreplace(labeldown, 7);
labelleft = zeroreplace(labelleft, 7);
labelright = zeroreplace(labelright, 7);
labelpush = zeroreplace(labelpush, 7);
labelpull = zeroreplace(labelpull, 7);

targets = vertcat(labeldown,labelup,labelleft,labelright,labelpush,labelpull);

% RMS
for mus = 1:14
    rms_down(:,mus)= signal_rms(down_data(:,mus), dsr, 0, 0);
    rms_up(:,mus)= signal_rms(up_data(:,mus), dsr, 0, 0);
    rms_left(:,mus)= signal_rms(left_data(:,mus), dsr, 0, 0);
    rms_right(:,mus)= signal_rms(right_data(:,mus), dsr, 0, 0);
    rms_push(:,mus)= signal_rms(push_data(:,mus), dsr, 0, 0);
    rms_pull(:,mus)= signal_rms(pull_data(:,mus), dsr, 0, 0);
end

inputs = vertcat(rms_down,rms_up,rms_left,rms_right,rms_push,rms_pull);

targets = downsample(targets, dsr);

n = size(inputs, 1);                % number of samples in the dataset
targetsd = dummyvar(targets);       % convert label into a dummy variable

inputs = inputs';                   % transpose input
targets = targets';                 % transpose target
targetsd = targetsd';               % transpose dummy variable

rng(1);                             % for reproducibility
c = cvpartition(n,'Holdout',1038);   % hold out 1/3 of the dataset

Xtrain = inputs(:, training(c));    % 2/3 of the input for training
Ytrain = targets(:, training(c));  % 2/3 of the target for training
Xtest = inputs(:, test(c));         % 1/3 of the input for testing
Ytest = targets(test(c));           % 1/3 of the target for testing
Ytestd = targetsd(:, test(c));      % 1/3 of the dummy variable for testing
Ytraind = targetsd(:,training(c));

Xtrain = Xtrain';
Ytrain = Ytrain';
Xtest = Xtest';
Ytest = Ytest';

rng default
svm4 = fitcecoc(Xtrain,Ytrain,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus','MaxObjectiveEvaluations',80))


predictedlabels = predict(svm4, Xtrain);
h=0; % displays accuracy of model using its predicted labels
j = 1;
while j <= length(predictedlabels)
    if predictedlabels(j) == Ytrain(j)
        h = h + 1;       
    end
     j = j + 1;
end
disp(h/length(predictedlabels));

predictedlabels2 = predict(svm4, Xtest);
m=0; % displays accuracy of model using its predicted labels
k = 1;
while k <= length(predictedlabels2)
    if predictedlabels2(k) == Ytest(k)
        m = m + 1;       
    end
     k = k + 1;
end
disp(m/length(predictedlabels2));
c1 = confusionmat(predictedlabels,Ytrain);
c2 = confusionmat(predictedlabels2,Ytest);
c1
c2

net2 = patternnet(20) %trying new sizes for pattern feedforward NN
net2.trainFcn = 'trainrp' %not the default, many options exist
net2.trainParam.max_fail = 50 %high but prevents sudden stops when trying new training functions

train(net2, Xtrain',Ytraind)
