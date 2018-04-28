load('database.mat');
M=nanmean(Agetrain);
Agetrain(isnan(Agetrain))=M;
Sextrain=Sextrain-1;
N=nanmean(Agetest);
K=nanmean(Faretest);
Agetest(isnan(Agetest))=N;
Faretest(isnan(Faretest))=K;
train=[Sextrain Agetrain SibSptrain Parchtrain Faretrain Pclasstrain  ];%sample用于训练分类器
test=[Sextest Agetest SibSptest Parchtest Faretest Pclasstest];%测试集
[m,n]=size(train);
for i=1:6
    %     train(:,i)=train(:,i)/norm(train(:,i));%归一化
    train(:,i)=(train(:,i)-min(train(:,i)))/(max(train(:,i))-min(train(:,i)));%归一化
end
for i=1:6
    test(:,i)=(test(:,i)-min(test(:,i)))/(max(test(:,i))-min(test(:,i)));%归一化
end

nb=NaiveBayes.fit(train,Survivedtrain);

predict_label1=predict(nb,train);
accuracy1=sum(predict_label1==Survivedtrain)/length(predict_label1);
fprintf('nb_train:accurace=%f\n',accuracy1);

predict_label1=predict(nb,test);
accuracy1=sum(predict_label1==Survivedtest)/length(predict_label1);
fprintf('nb_test:accurace=%f\n',accuracy1);

for sigma=0.01:1:5.01
    svmModel=svmtrain(train,Survivedtrain,'kernel_function','rbf','rbf_sigma',...
        sigma,'showplot',false);
    
    predict_label2=svmclassify(svmModel,train);
    accuracy2=sum(predict_label2==Survivedtrain)/length(predict_label2);
    fprintf('svm_train(sigma=%f):accurace=%f\n',sigma,accuracy2);
end
for sigma=0.01:1:5.01
    svmModel=svmtrain(train,Survivedtrain,'kernel_function','rbf','rbf_sigma',...
        sigma,'showplot',false);
    
    predict_label2=svmclassify(svmModel,test);
    accuracy2=sum(predict_label2==Survivedtest)/length(predict_label2);
    fprintf('svm_test(sigma=%f):accurace=%f\n',sigma,accuracy2);
end
% train=[train(:,1) train(:,6)];
% test=[test(:,1) test(:,6)];
    svmModel=svmtrain(train,Survivedtrain,'kernel_function','linear','showplot',false);

    predict_label2=svmclassify(svmModel,test);
    accuracy2=sum(predict_label2==Survivedtest)/length(predict_label2);
    fprintf('svm_test(linear):accurace=%f\n',sigma,accuracy2);
     svmModel=svmtrain(train,Survivedtrain,'kernel_function','linear','showplot',false);

    predict_label2=svmclassify(svmModel,train);
    accuracy2=sum(predict_label2==Survivedtrain)/length(predict_label2);
    fprintf('svm_train(linear):accurace=%f\n',sigma,accuracy2);