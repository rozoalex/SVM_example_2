clear;

train_data = importdata('train_s1.csv');

number_of_training = 18500;

train = train_data(1:number_of_training,:);
test = train_data((number_of_training+1):end,:);

SVMModel = fitcsvm(train(:,2:end),train(:,1),'KernelFunction','rbf','Cost',[0 8000; 5000 0]);

%,'KernelFunction','rbf','Standardize',true,'ClassNames',{'negClass','posClass'}

[label,score] = predict(SVMModel,test(:,2:end));

CVSVMModel = crossval(SVMModel);

classLoss = kfoldLoss(CVSVMModel);

% plot(test(:,1),label);

correct_rate = 1 - length(find(test(:,1)~=label))/length(label);

