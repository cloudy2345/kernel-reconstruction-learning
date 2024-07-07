function [x_train,x_test,x_node,fz]=shuju5()  %密度估计
n_train=600;  % 各100个
n_test=10000;
nsuiji=5000;
m=8;
x_dim=1;
alpha=0.4;
%----------------------------------产生训练集--------------------------------------------
u1=-2 ;
u2=1;
v1=0.5;
v2=0.5;  %标准差

x_train=zeros(n_train,x_dim);   %混合高斯分布
for i=1:n_train
    if rand(1)<alpha
        x_train(i,:)=normrnd ( u1, v1,1 );
    else 
        x_train(i,:)=normrnd ( u2, v2, 1 );
    end
end


%----------------------产生测试集—mcmc—————————————————————
x_test=zeros(n_test,x_dim);   %混合高斯分布
for i=1:n_test
    if rand(1)<alpha
        x_test(i,:)=normrnd ( u1, v1,1 );
    else 
        x_test(i,:)=normrnd ( u2, v2, 1 );
    end
end

%----------------------------找节点---------------------------------------
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
        a=randperm(n_train/2);
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
%x_node=[3,-5,-1,0,-3]';


fz=@(x) alpha.*(2.*pi).^(-x_dim/2).*(det(v1.^2))^(-0.5)*exp(-(x'-u1)'*inv(v1.^2)*(x'-u1)./2)+(1-alpha).*(2.*pi).^(-x_dim/2).*(det(v2.^2))^(-0.5)*exp(-(x'-u2)'*inv(v2.^2)*(x'-u2)./2);

end