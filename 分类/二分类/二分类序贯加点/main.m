function [a,b,c,d,e,f,cishu]=main(nchangshi)  %a,b原来的  c,d添加后的
a=0;b=0;c=0;d=0;zz=0;e=0;f=0;
for k=1:nchangshi
    try
        [x_train,x_test,y_train,y_test,x_node,y_node,node]=shuju6();
        node2=xunlian8(x_train,y_train,x_node,y_node,node)  ; %d方法加点
        node3=xunlian7(x_train,y_train,x_node,y_node,node) ;  %p加点法
        x_node2=x_train(node2,:);
        y_node2=y_train(node2);
        x_node3=x_train(node3,:);
        y_node3=y_train(node3);
        fenlei1=xunlian2(x_train,y_train,x_node,y_node)  ; %不加点模型
        fenlei2=xunlian3(x_train,y_train,x_node2);       %d
        fenlei3=xunlian2(x_train,y_train,x_node3,y_node3);   %p
        [error_train1,error_test1]=ceshi(fenlei1,x_train,x_test,y_train,y_test);
        [error_train2,error_test2]=ceshi(fenlei2,x_train,x_test,y_train,y_test);
        [error_train3,error_test3]=ceshi(fenlei3,x_train,x_test,y_train,y_test);
        
 %       if error_test3>0.30
 %           error('nuyao')
 %       end    
        a=a+error_train1;
        b=b+error_test1;
        c=c+error_train2;
        d=d+error_test2;
        e=e+error_train3;
        f=f+error_test3;
    catch
        zz=zz+1
    end

end
cishu=nchangshi-zz;
a=a/cishu;
b=b/cishu;
c=c/cishu;
d=d/cishu;
e=e/cishu;
f=f/cishu;

end
