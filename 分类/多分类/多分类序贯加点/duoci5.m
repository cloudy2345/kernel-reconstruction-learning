function [a,b,c,cishu]=duoci5(nchangshi)   %�����P�㷨���лع��߸���ͬ��
a=0;b=0;c=0;zz=0;d=0;e=0;
ll1=ones(nchangshi,1);

for k=1:nchangshi
    try
        [x_train1,x_train2,x_train3,x_train,y_train,x_test,y_test,x_node1,x_node2,x_node3,x_node,node]=shuju11()  ;
        nn1=length(y_test);
     

        node2=xunlian7(x_train,y_train,x_node,node) ;        %�лع���
        moxing3=xunlian2(x_train,y_train,x_train(node2,:))
        aa3=zeros(nn1,1);
        for i=1:nn1
            aa3(i)=moxing3(x_test(i,:));
        end
        logistic1=sum(y_test~=aa3)./nn1;
        disp(['logistic1=' num2str(logistic1) ]);
   
        
        node3=xunlian8(x_train,y_train,x_node,node)          %�޻ع���
        moxing33=xunlian1(x_train,y_train,x_train(node3,:));
        aa33=zeros(nn1,1);
        for i=1:nn1
            aa33(i)=moxing33(x_test(i,:));
       end
        logistic2=sum(y_test~=aa33)./nn1;
        disp(['logistic2=' num2str(logistic2) ]);
  
        
        
       
        
        ll1(k)=logistic1;

        a=a+logistic1;
        b=b+logistic2;


    catch
        zz=zz+1
    end
end
cishu=nchangshi-zz
a=a/cishu   %��
b=b/cishu   %��
c=c/cishu   %svm
std(ll1(ll1~=1))


end