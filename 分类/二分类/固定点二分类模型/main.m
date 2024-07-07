function [a,b,c,d,cishu]=main(nchangshi)
a=0;b=0;c=0;d=0;zz=0;
for k=1:nchangshi
    try
        [x_train,x_test,y_train,y_test,x_node,y_node]=shuju6();
        fenlei=model2(x_train,y_train,x_node,y_node)  ; %·µ»Ø·ÖÀàÄ£ÐÍ
        [error_train,error_test,svm_error_train,svm_error_test]=ceshi(fenlei,x_train,x_test,y_train,y_test);

        a=a+error_train;
        b=b+error_test;
        c=c+svm_error_train;
        d=d+svm_error_test;
    catch
        zz=zz+1
    end

end
cishu=nchangshi-zz;
a=a/cishu;
b=b/cishu;
c=c/cishu;
d=d/cishu;

end
