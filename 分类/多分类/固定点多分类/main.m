function [a,b,c,cishu]=main(nchangshi)   %多分类模型，SVM与KLR
a=0;b=0;c=0;zz=0;d=0;e=0;
ll1=ones(nchangshi,1);
ll11=ll1;
ll2=ll1;
ll22=ll1;
ll3=ll1;

for k=1:nchangshi
    try
        [x_train1,x_train2,x_train3,x_train,y_train,x_test,y_test,x_node,node]=dataset13()  ;
        nn1=length(y_test);
     
        moxing1=model1(x_train,y_train,x_node); %无回归项,各项同性
        aa1=zeros(nn1,1);
        for i=1:nn1
            aa1(i)=moxing1(x_test(i,:));
        end
        wuhui1=sum(y_test~=aa1)./nn1;
        disp(['wuhui1=' num2str(wuhui1) ])
        
        moxing11=model11(x_train,y_train,x_node); %无回归项，各向异性
        aa11=zeros(nn1,1);
        for i=1:nn1
            aa11(i)=moxing11(x_test(i,:));
        end
        wuhui2=sum(y_test~=aa11)./nn1;
        disp(['wuhui2=' num2str(wuhui2) ])
        
        moxing2=model2(x_train,y_train,x_node);%有回归项，各项同性
        aa2=zeros(nn1,1);
        for i=1:nn1
            aa2(i)=moxing2(x_test(i,:));
        end
        youhui1=sum(y_test~=aa2)./nn1;
        disp(['youhui1=' num2str(youhui1) ])
        
        moxing22=model22(x_train,y_train,x_node);%有回归项，各项异性
        aa22=zeros(nn1,1);
        for i=1:nn1
            aa22(i)=moxing22(x_test(i,:));
        end
        youhui2=sum(y_test~=aa22)./nn1;
        disp(['youhui2=' num2str(youhui2) ])
     
        moxing3=model3(x_train,y_train,x_train) ; %KLR
        aa3=zeros(nn1,1);
        for i=1:nn1
            aa3(i)=moxing3(x_test(i,:));
        end
        klr3=sum(y_test~=aa3)./nn1;
        disp(['klr3=' num2str(klr3) ])

     
      %  node2=model7(x_train,y_train,x_node,node) ;        %有回归项
     %   moxing3=model2(x_train,y_train,x_train(node2,:))
      %  aa3=zeros(nn1,1);
      %  for i=1:nn1
      %      aa3(i)=moxing3(x_test(i,:));
      %  end
      %  logistic1=sum(y_test~=aa3)./nn1;
      %  disp(['logistic1=' num2str(logistic1) ]);
   
        
     %   node3=model8(x_train,y_train,x_node,node)          %无回归项
     %   moxing33=model1(x_train,y_train,x_train(node3,:));
     %   aa33=zeros(nn1,1);
    %    for i=1:nn1
    %        aa33(i)=moxing33(x_test(i,:));
    %    end
     %   logistic2=sum(y_test~=aa33)./nn1;
   %     disp(['logistic2=' num2str(logistic2) ]);
  
        
        
        t = templateSVM('Standardize',true,'KernelFunction','gaussian', 'KernelScale','auto');
        SVMmodel = fitcecoc(x_train,y_train,'Learners',t);
        yy1=predict(SVMmodel,x_test);
        svm_error_test=sum(yy1~=y_test)/nn1;
        disp(['svm-test_error=' num2str(svm_error_test) ])
       
        
        ll1(k)=wuhui1;
        ll11(k)=wuhui2;
        ll2(k)=youhui1;
        ll22(k)=youhui2;
        ll3(k)=klr3;
        ll4(k)=svm_error_test;
     %   a=a+logistic1;
      %  b=b+logistic2;
        b=0;
        c=c+svm_error_test;

    catch
        zz=zz+1
    end
end
cishu=nchangshi-zz;
a=a/cishu;   %有
b=b/cishu;   %无
c=c/cishu;   %svm
std(ll1(ll1~=1))
std(ll11(ll11~=1))
std(ll2(ll2~=1))
std(ll22(ll22~=1))
std(ll3(ll3~=1))  %klr
std(ll4(ll4~=1))  %sm


