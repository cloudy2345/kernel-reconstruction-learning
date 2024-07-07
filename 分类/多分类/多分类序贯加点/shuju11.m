function [x_train1,x_train2,x_train3,x_train,y_train,x_test,y_test,x_node1,x_node2,x_node3,x_node,node]=shuju11() %分类
n_train=450;  % 各100个
n_test=9000;
nsuiji=5000;
m=16;  %密度估计
x_dim=2;
m2=8;  %KLR
%----------------------------------产生训练集--------------------------------------------



u1=[1;0];
u2=[0;1];
u3=[0;0];
v1=[1,-1;-1,2]./8;
v2=[2,-1;-1,1]./8;  %协方差矩阵
v3=[1,0;0,2]./8;
x_train1=mvnrnd ( u1, v1, n_train./3 );   %100行2列，每行一个x
x_train2=mvnrnd ( u2, v2, n_train./3 );
x_train3=mvnrnd ( u3, v3, n_train./3 );
x_train=[x_train1;x_train2;x_train3];
y_train=[ones(n_train/3,1);2.*ones(n_train/3,1);3.*ones(n_train/3,1)];

%----------------------产生测试集——————————————————————
X11=mvnrnd ( u1, v1, n_test./3 );   %100行2列，每行一个x
X22=mvnrnd ( u2, v2, n_test./3 );
X33=mvnrnd ( u3, v3, n_test./3 );
x_test=[X11;X22;X33];
y_test=[ones(n_test/3,1);2.*ones(n_test/3,1);3.*ones(n_test/3,1)];


function zuida=juli(X)   %一个m个点的集合的内距离
    zuida=0;
    for i=1:m-1
        for j=i+1:m
            a=sum(1./abs(X(i,:)-X(j,:)));
            if a>zuida
                zuida=a;
            end
        end
    end
end
function minnode=jiedian(train)  %从x_train中找到最好的m个点
    zuid=10^10;
    for i=1:nsuiji
        a=randperm(n_train/3);
        weizhi=a(1:m);
        Y=juli(train(weizhi,:));
        if Y<zuid
            zuid=Y;
            minnode=weizhi;
        end
    end
end 
node1=jiedian(x_train1);
x_node1=x_train1(node1,:);
node2=jiedian(x_train2);
x_node2=x_train2(node2,:);
node3=jiedian(x_train3);
x_node3=x_train2(node3,:);

%---------------------------------重构选用的节点--------------------------

function zuida=juli2(X)   %一个m个点的集合的内距离
    zuida=0;
    for i=1:m2-1
        for j=i+1:m2
            a=sum(1./abs(X(i,:)-X(j,:)));
            if a>zuida
                zuida=a;
            end
        end
    end
end
function minnode=jiedian2(train)  %从x_train中找到最好的m个点
    zuid=10^10;
    for i=1:nsuiji
        a=randperm(n_train);
        weizhi=a(1:m2);
        Y=juli2(train(weizhi,:));
        if Y<zuid
            zuid=Y;
            minnode=weizhi;
        end
    end
end 
node=jiedian2(x_train);
x_node=x_train(node,:);




end