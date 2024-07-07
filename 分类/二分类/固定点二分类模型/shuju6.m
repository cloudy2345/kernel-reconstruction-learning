function [x_train,x_test,y_train,y_test,x_node,y_node]=shuju6()
n_train=200;
n_test=10000;
nsuiji=1000;
m=20;
x_dim=2;
%----------------------------------产生训练集--------------------------------------------
u1=[1;0];
u2=[0;1];
uk1=mvnrnd ( u1, eye(x_dim), 10 );   %10行2列，每行一个uk
uk2=mvnrnd ( u2, eye(x_dim), 10 );

function  yige=yige(uk1)   %产生一个x
    k = randi([1,10],1) ; %各占0.1概率;
    yige=mvnrnd ( uk1(k,:), 0.5.*eye(x_dim), 1 ) ;  %行向量
end

x_train=zeros(n_train,x_dim);
for i=1:n_train/2
    x_train(i,:)=yige(uk1);        %前一半1，后一半-1
end
for i=n_train/2+1:n_train
    x_train(i,:)=yige(uk2);
end
y_train=[ones(n_train/2,1);-ones(n_train/2,1)];
%----------------------------------产生训练集--------------------------------------------
x_test=zeros(n_test,x_dim);
for i=1:n_test/2
    x_test(i,:)=yige(uk1);
end
for i=n_test/2+1:n_test
    x_test(i,:)=yige(uk2);
end
y_test=[ones(n_test/2,1);-ones(n_test/2,1)];

%-----------------------------------标准化————————————————————
train_mean=mean(x_train);
train_std=std(x_train);
x_train=(x_train-train_mean)./train_std;
x_test=(x_test-train_mean)./train_std;

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
        a=randperm(n_train);
        weizhi=a(1:m);
        Y=juli(train(weizhi,:));
        if Y<zuid
            zuid=Y;
            minnode=weizhi;
        end
    end
end 
node=jiedian(x_train);
x_node=x_train(node,:);
y_node=y_train(node,:);


end