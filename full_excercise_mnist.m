clearvars;
close all;
clc;

%% Load Data and Display Some part of it.

fprintf('Press enter to load data and train the machine. It will only take a few minutes...\n');
pause();
fprintf('Loading data and training the machine...\n');

data = load('mnist_train.csv');
X = data(:,2:785);
y = data(:,1);
len = size(X,1);

random_index = randperm(len);
X = X(random_index,:);
y = y(random_index);

data2 = load('mnist_test.csv');
rand_indx = randperm(length(data2));
selected_examples = data2(rand_indx,2:785);
selected_examples_y = data2(rand_indx,1);

%displayData(selected_examples);
%X = X(1001:len,:);
%y = y(1001:len,:);

%% Finding optimal solution of thetas.

all_theta = zeros(10,785);
lambda = 0.1;
initial_theta = zeros(785,1);
X = [ones(size(X,1),1) X];
selected_examples = [ones(size(selected_examples,1),1) selected_examples];

options = optimset('GradObj', 'on', 'MaxIter', 50);

for i = 1:10
    y_curr = y==i;
    %all_theta(i,:) = pinv(X'*X)*X'*y_curr;
    all_theta(i,:) = fmincg(@(t)(lr_cost(X, y==mod(i,10), t, lambda)),initial_theta,options);
end

corr_predict = 0;
fprintf('Machine is trained.\n');
fprintf('You can now look at the results of 100 randomly selected test examples,\n');
fprintf('while the machine calculates its accuracy over 10,000 test examples...\n');
fprintf('Press enter to continue...\n');
pause();
fprintf('Press enter everytime you want to move to the next example...\n');
for i = 1:size(selected_examples,1)
    prediction = all_theta*selected_examples(i,:)';
    [~, index] = max(prediction);
    if mod(i,100) == 0
        colormap(gray);
        imagesc(reshape(selected_examples(i,2:785),28,28)');
        title(['Example : ',int2str(i/100),'   Prediction : ',int2str(mod(index,10)),'   Correct Answer : ',int2str(mod(selected_examples_y(i),10))]);
    end
    if(mod(index,10) == selected_examples_y(i))
        corr_predict = corr_predict + 1;
    end
    if mod(i,100) == 0
        pause();
    end
end
disp("Accuracy of the programme : "+corr_predict/100+"%");