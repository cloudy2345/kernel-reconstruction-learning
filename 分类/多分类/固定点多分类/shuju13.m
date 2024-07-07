function [x_train1,x_train2,x_train3,x_train,y_train,x_test,y_test,x_node1,x_node2,x_node3,x_node,node]=shuju13() %分类
n_train=600;  % 各200个
n_test=9000;
nsuiji=5000;
m=24;
x_dim=3;
m2=12;
%----------------------------------产生训练集--------------------------------------------

u11=[1,0,0];
u12=[0.8,0,0.1];
u21=[0,0,1];
u22=[0.1,0,1.2];
u31=[0,1,0];
u32=[0.2,1,0];
v1=[1,-1,1;-1,2,-1;1,-1,2]./4;
v2=[2,-1,1;-1,2,-1;1,-1,2]./4;  %协方差矩阵
v3=eye(3)./9;
x_train1=zeros(n_train/3,3);
for i1=1:n_train/3
    if rand(1)<0.5
        x_train1(i1,:)=mvnrnd ( u11, v1, 1 );
    else
        x_train1(i1,:)=mvnrnd ( u12, v1, 1 );
    end
end

x_train2=zeros(n_train/3,3);
for i2=1:n_train/3
    if rand(1)<0.5
        x_train2(i2,:)=mvnrnd ( u21, v2, 1 );
    else
        x_train2(i2,:)=mvnrnd ( u22, v2, 1 );
    end
end

x_train3=zeros(n_train/3,3);
for i3=1:n_train/3
    if rand(1)<0.5
        x_train3(i3,:)=mvnrnd ( u31, v3, 1 );
    else
        x_train3(i3,:)=mvnrnd ( u32, v3, 1 );
    end
end

x_train=[x_train1;x_train2;x_train3];
y_train=[ones(n_train/3,1);2.*ones(n_train/3,1);3.*ones(n_train/3,1)];

%----------------------产生测试集——————————————————————
x_test1=zeros(n_test/3,3);
for i1=1:n_test/3
    if rand(1)<0.5
        x_test1(i1,:)=mvnrnd ( u11, v1, 1 );
    else
        x_test1(i1,:)=mvnrnd ( u12, v1, 1 );
    end
end

x_test2=zeros(n_test/3,3);
for i2=1:n_test/3
    if rand(1)<0.5
        x_test2(i2,:)=mvnrnd ( u21, v2, 1 );
    else
        x_test2(i2,:)=mvnrnd ( u22, v2, 1 );
    end
end

x_test3=zeros(n_test/3,3);
for i3=1:n_test/3
    if rand(1)<0.5
        x_test3(i3,:)=mvnrnd ( u31, v3, 1 );
    else
        x_test3(i3,:)=mvnrnd ( u32, v3, 1 );
    end
end
x_test=[x_test1;x_test2;x_test3];
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