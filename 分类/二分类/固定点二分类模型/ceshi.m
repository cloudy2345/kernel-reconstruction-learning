%------------------------------º∆À„ŒÛ≤Ó---------------------------------
function [error_train,error_test,svm_error_train,svm_error_test]=ceshi(fenlei,x_train,x_test,y_train,y_test)
size1=size(x_train);
n_train=size1(1);
size2=size(x_test);
n_test=size2(1);
cuo=0;
for i=1:n_train
    if y_train(i)~=fenlei(x_train(i,:));
        cuo=cuo+1;
    end
end
error_train=cuo/n_train;
disp(['train_error=' num2str(error_train) ])
cuo=0;
for i=1:n_test
    if y_test(i)~=fenlei(x_test(i,:));
        cuo=cuo+1;
    end
end
error_test=cuo/n_test;
disp(['test_error=' num2str(error_test) ])

%---------------------svm---------------------------------
SVMModel = fitcsvm(x_train,y_train,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');

cuo=0;
for i=1:n_train
    if y_train(i)~=predict(SVMModel,x_train(i,:));
        cuo=cuo+1;
    end
end
svm_error_train=cuo/n_train;
disp(['svm-train_error=' num2str(svm_error_train) ])
cuo=0;
for i=1:n_test
    if y_test(i)~=predict(SVMModel,x_test(i,:));
        cuo=cuo+1;
    end
end
svm_error_test=cuo/n_test;
disp(['svm-test_error=' num2str(svm_error_test) ])

end
