%this involves movingmean then downsampling to reduce number of datapoints,
%currently sampled at 1925.9 Hz, downsamples to 19.259 Hz (factor of 100)
close all; clear all; tic;
dsr = 80; %downsampling rate, can use in RMS
dsro = (dsr - 1); %offset to keep label and input matrices same size
fs = 1925.9;
fc = 6; % cut-off frequency (~from 4-10Hz)
N = 3; % filter order
[B, A] = butter(N,fc*2/fs,'low'); % low-pass Butterworth filter parameters
c1 = 8.443e-5; %cutoffs for training data labels
c2 = 0.0001282;
c3 = 7.285e-5;
c4 = 9.601e-5;
c5 = 9.098e-5;
c6 = 7.822e-5;

INPUT_DATA = csvread('DOWN_Plot_and_Store_Combined_Train.csv',0,0) ;
output_down = downsample(label2(load_data(INPUT_DATA),1,c1),dsr,dsro);
input_down = matrix_rms(filtfilt(B, A, load_data(INPUT_DATA)),dsr,0,0);

INPUT_DATA = csvread('UP_Plot_and_Store_Combined_Train.csv',0,0) ;
output_up = downsample(label2(load_data(INPUT_DATA),2,c2),dsr,dsro);
input_up = matrix_rms(filtfilt(B, A, load_data(INPUT_DATA)),dsr,0,0);

INPUT_DATA = csvread('LEFT_Plot_and_Store_Combined_Train.csv',0,0) ;
output_left = downsample(label2(load_data(INPUT_DATA),3,c3),dsr,dsro);
input_left = matrix_rms(filtfilt(B, A, load_data(INPUT_DATA)),dsr,0,0);

INPUT_DATA = csvread('RIGHT_Plot_and_Store_Combined_Train.csv',0,0) ;
output_right = downsample(label2(load_data(INPUT_DATA),4,c4),dsr,dsro);
input_right = matrix_rms(filtfilt(B, A, load_data(INPUT_DATA)),dsr,0,0);

INPUT_DATA = csvread('PUSH_Plot_and_Store_Combined_Train.csv',0,0) ;
output_push = downsample(label2(load_data(INPUT_DATA),5,c5),dsr,dsro);
input_push = matrix_rms(filtfilt(B, A, load_data(INPUT_DATA)),dsr,0,0);

INPUT_DATA = csvread('PULL_Plot_and_Store_Combined_Train.csv',0,0) ;
output_pull = downsample(label2(load_data(INPUT_DATA),6,c6),dsr,dsro);
input_pull = matrix_rms(filtfilt(B, A, load_data(INPUT_DATA)),dsr,0,0);
% 
% output_down = label_data(input_down,1);
% output_up = label_data(input_up,2);
% output_left = label_data(input_left,3);
% output_right = label_data(input_right,4);
% output_push = label_data(input_push,5);
% output_pull = label_data(input_pull,6);
%       
data_labels_output_train = vertcat(output_down, output_up, output_left, output_right, output_push, output_pull);
combined_data_subj1_input_train = vertcat(input_down, input_up, input_left, input_right, input_push, input_pull);



INPUT_DATA = csvread('DOWN_Plot_and_Store_Combined_Test.csv',0,0) ;
output_down = downsample(label2(load_data(INPUT_DATA),1,c1),dsr,dsro);
input_down =  matrix_rms(filtfilt(B, A, load_data(INPUT_DATA)),dsr,0,0);

INPUT_DATA = csvread('UP_Plot_and_Store_Combined_Test.csv',0,0) ;
output_up = downsample(label2(load_data(INPUT_DATA),2,c2),dsr,dsro);
input_up =  matrix_rms(filtfilt(B, A, load_data(INPUT_DATA)),dsr,0,0);

INPUT_DATA = csvread('LEFT_Plot_and_Store_Combined_Test.csv',0,0) ;
output_left = downsample(label2(load_data(INPUT_DATA),3,c3),dsr,dsro);
input_left = matrix_rms(filtfilt(B, A, load_data(INPUT_DATA)),dsr,0,0);

INPUT_DATA = csvread('RIGHT_Plot_and_Store_Combined_Test.csv',0,0) ;
output_right = downsample(label2(load_data(INPUT_DATA),4,c4),dsr,dsro);
input_right = matrix_rms(filtfilt(B, A, load_data(INPUT_DATA)),dsr,0,0);

INPUT_DATA = csvread('PUSH_Plot_and_Store_Combined_Test.csv',0,0) ;
output_push = downsample(label2(load_data(INPUT_DATA),5,c5),dsr,dsro);
input_push =  matrix_rms(filtfilt(B, A, load_data(INPUT_DATA)),dsr,0,0);

INPUT_DATA = csvread('PULL_Plot_and_Store_Combined_Test.csv',0,0) ;
output_pull = downsample(label2(load_data(INPUT_DATA),6,c6),dsr,dsro);
input_pull = matrix_rms(filtfilt(B, A, load_data(INPUT_DATA)),dsr,0,0);
% 
% output_down = label_data(input_down,1);
% output_up = label_data(input_up,2);
% output_left = label_data(input_left,3);
% output_right = label_data(input_right,4);
% output_push = label_data(input_push,5);
% output_pull = label_data(input_pull,6);
%       
data_labels_output_test = vertcat(output_down, output_up, output_left, output_right, output_push, output_pull);
combined_data_subj1_input_test = vertcat(input_down, input_up, input_left, input_right, input_push, input_pull);

% Low-pass Filtering

% application of filtering




%trainingoptions('lm' 
% 
% cnn_labels = categorical(data_labels_output_train);
% cnn_data = reshape(combined_data_subj1_input_train, [14,1,1,17251]);      
% cnn_labels_test = categorical(data_labels_output_test);
% cnn_data_test = reshape(combined_data_subj1_input_test, [14,1,1,5726]);      
%                     
% options = trainingOptions('sgdm', ...
%     'InitialLearnRate',0.1, ...
%     'MaxEpochs',100, ...
%     'Shuffle','every-epoch', ...
%     'Verbose',false, ...
%     'Plots','training-progress', ...
%     'ValidationData', {cnn_data_test,cnn_labels_test});
% layers = [imageInputLayer([14 1 1], 'Normalization', 'none')
%           convolution2dLayer([1 1],16,'stride',1)    
%           reluLayer
%           maxPooling2dLayer([1 1],'stride',1)
%           fullyConnectedLayer(6)
%           softmaxLayer
%           classificationLayer];
% 
% cnn4 = trainNetwork(cnn_data,cnn_labels,layers,options);

% 
% output_down = label2(input_down,1); %label3 gives 7 for no movement instead of 0, so can be classified in patternnet 
% output_up = label2(input_up,2);
% output_left = label2(input_left,3);
% output_right = label2(input_right,4);
% output_push = label2(input_push,5);
% output_pull = label2(input_pull,6);

combined_data_subj1_input_t = combined_data_subj1_input_train.';
data_labels_output_diag = vertcat(output_down, output_up, output_left, output_right, output_push, output_pull);
data_labels_output_diag_t = data_labels_output_diag.';
% 
% pnet3 = feedforwardnet(10);
% pnet3.divideFcn = 'divideblock';  
% pnet3.trainParam.max_fail = 15;
% [pnet_3, tr2] = train(pnet3, combined_data_subj1_input_t, data_labels_output_diag_t);


rng default
svm3 = fitcecoc(combined_data_subj1_input_train,data_labels_output_train,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus','MaxObjectiveEvaluations',80))

% 
% t3 = templateSVM('SaveSupportVectors','on');
% svm3 = fitcecoc(combined_data_subj1_input_train,data_labels_output_train,'Learners',t3);

predictedlabels = predict(svm3, combined_data_subj1_input_train);
h=0; % displays accuracy of model using its predicted labels
j = 1;
while j <= length(predictedlabels)
    if predictedlabels(j) == data_labels_output_train(j)
        h = h + 1;       
    end
     j = j + 1;
end
disp(h/length(predictedlabels));

predictedlabels2 = predict(svm3, combined_data_subj1_input_test);
m=0; % displays accuracy of model using its predicted labels
k = 1;
while k <= length(predictedlabels2)
    if predictedlabels2(k) == data_labels_output_test(k)
        m = m + 1;       
    end
     k = k + 1;
end
disp(m/length(predictedlabels2));
c1 = confusionmat(predictedlabels,data_labels_output_train);
c2 = confusionmat(predictedlabels2,data_labels_output_test);
c1
c2

% 
% tdn1 = timedelaynet(1:300, 10) %seems to be meant for 1D data
% tdn1net = train(tdn1, combined_data_subj1_input_train.', data_labels_output_train.');

% 
% data_input_train_t = combined_data_subj1_input_train.' 
% data_output_train_t = data_labels_output_train.'
% 
% var = 7;
% data_labels_output_train2 = data_labels_output_train.';
% T = num2cell(data_labels_output_train2(1:end));
% delaySize = 75;
% hiddenSizes = 7;
% combined_data_subj1_input_train2 = combined_data_subj1_input_train.';
% for i = 1:7
%     net{i} = layrecnet(1:delaySize, hiddenSizes);
%     X = num2cell(combined_data_subj1_input_train2(i,1:end));
%     [Xs{i}, Xi{i},Ai{i},Ts{i}] = preparets(net{i},X,T);
%     trained_net{i} = train(net{i},Xs{i},Ts{i},Xi{i},Ai{i});,
% end
% for i = 1:7
%     y(i) = trained_net{i}(Xs{i},Xi{i},Ai{i});
%     perf(i) = perform(trained_net({i},y{i},Ts{i});
%     
% end
% perf = mean(perf)