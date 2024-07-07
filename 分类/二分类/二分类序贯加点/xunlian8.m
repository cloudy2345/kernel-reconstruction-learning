%无回归项 各向同性 hinge-loss   进行后续加点设计,根据距离加
function node2=xunlian8(x_train,y_train,x_node,y_node,node)   %返回分类模型
size1=size(x_train);
x_dim=size1(2);
n_train=size1(1);
size2=size(x_node);
m=size2(1);
lambda1=0.3;
theta0=0.4;
jia=3*x_dim;

%---------------------------kernel--------------------------------
function k=kernel(theta,x,y)         %行向量 theta>0
    k=exp(-1/x_dim.*sum((x-y).^2.*theta));
end

%--------------------------------------构建使用函数----------------------------------
function RA=RAC(theta)  %产生RA
    RA=eye(m);   %对角线是1
    %只算了上半部分
    for i=1:m-1
        for j=i+1:m
            RA(i,j)=kernel(theta,x_node(i,:),x_node(j,:));  
        end
    end
    RA=RA+RA'-eye(m)+10^(-6).*eye(m);
end
function r=rA(theta,x)   %ra(x)到结点的列向量,x行向量
    r=zeros(m,1);
    for i=1:m
        r(i)=kernel(theta,x,x_node(i,:));
    end
end
function rA_Z=rA_ZC(theta)
    rA_Z=zeros(m,n_train);
    for i=1:n_train
        rA_Z(:,i)=rA(theta,x_train(i,:));
    end
end
%--------------------------------SQP--------------------------------------
function [canshu,loss,exitflag]=SQP(theta)  %前面m是gamma，然后b,后面是t
    RA=RAC(theta);
    RAN=inv(RA)  ;  %把逆提出来节省时间
    H=zeros(m+n_train+1);
    H(1:m,1:m)=RAN   ;      %H
    f=ones(m+n_train+1,1);
    f(1:m+1)=0      ;           %f
    b=[zeros(n_train,1);-ones(n_train,1)] ;  %b
    A1=[zeros(n_train,m+1),-eye(n_train)];
    A2=zeros(n_train,m);
    for i=1:n_train
        A2(i,:)=-1.*y_train(i).*rA(theta,x_train(i,:))'*RAN;
    end
    A2=[A2,-1.*y_train,-eye(n_train)];
    A=[A1;A2]     ;       %A
    [canshu,loss,exitflag]=quadprog(lambda1.*H,f,A,b);
end
%----------------------------------loss--------------------------------- 
function loss=loss(theta,gamma1)  %(gamma1为gamma与b)列向量，theta行向量
    RA=RAC(theta);
    RAN=inv(RA)  ;  %把逆提出来节省时间
    loss=0;
    gamma=gamma1(1:m);
    b=gamma1(m+1);
    for i=1:n_train
        f=gamma'*RAN*rA(theta,x_train(i,:))+b;      %函数值
        loss=loss+max(1-y_train(i).*f,0);
    end
    loss=loss+0.5.*lambda1.*gamma'*RAN*gamma;
end
%---------------------------------估计参数--------------------------------
cha=100;
while cha>0.1 | cha <-0.3
    [canshu,loss1,exitflag1]=SQP(theta0) ;
    loss1
    gamma1=canshu(1:m+1);
  %  [theta0,loss2,exitflag2]=fmincon(@(theta) loss(theta,gamma1),theta0,[],[],[],[],0.0001.*ones(1,x_dim),[]);
   [theta0,loss2,exitflag2]=fmincon(@(theta) loss(theta,gamma1),1,[],[],[],[],0.0001,[]); %同性
    loss2
    cha=loss1-loss2;
end      %找到theta0与gamma


%----------------------------------------开始新加点--------------------------------------------
RA=RAC(theta0);
RAN=inv(RA)  ;  %把逆提出来节省时间

Zong=1:n_train;
S=setdiff(Zong,node);  %备选集

m=length(node);
x_node=x_train(node,:);
K_1=zeros(n_train,m);
for i13=1:n_train
    b=RAN*rA(theta0,x_train(i13,:));
    K_1(i13,:)=b';
end
K_1=[K_1,ones(n_train,1)];     %最后一列是1

P=y_train.*(K_1*gamma1);   %这个就是距离
P_yu=P(S);  %在剩余点的p

KK=0;


while true
    [min1,min_weizhi]=min(P_yu);  %最小值于最小值的位置 
    min1
    node_jia=S(min_weizhi);
    m=m+1;
    node(m)=node_jia;        %添加进来最新的
    x_node(m,:)=x_train(node_jia,:);
    y_node(m)=y_train(node_jia);
    
    for i17=1:m-1
        RA(i17,m)=kernel(theta0,x_train(node_jia,:),x_node(i17));
        RA(m,i17)=RA(i17,m);
    end
    RA(m,m)=1;    %新的RA
    RAN=inv(RA)  ;  %把逆提出来节省时间
    K_1=zeros(n_train,m);
    for i13=1:n_train
        b=RAN*rA(theta0,x_train(i13,:));
        K_1(i13,:)=b';
    end
    K_1=[K_1,ones(n_train,1)];
    
    [canshu,loss1,exitflag1]=SQP(theta0) ;
    gamma1=canshu(1:m+1);               %再次估计gamma
    P=y_train.*(K_1*gamma1);   %这个就是距离
    S=setdiff(Zong,node);
    P_yu=P(S);  %在剩余点的p
    
    %if abs(H11-H_old)/H_old<0.001 | KK>20
    KK=KK+1
   % if KK>=5 | min1>0.9
    if KK>=jia
        break
    end
   
    if mod(KK,5)==0  %5步冲估计一次theta
         [theta0,loss2,exitflag2]=fmincon(@(theta) loss(theta,gamma1),1,[],[],[],[],0.0001,[]);  %同性 
    end
   %if min1>-0.8  %没有分错

    
end

%------------------------------计算误差---------------------------------

x_node=x_train(node,:);
m=length(node);
RA=RAC(theta0);
RAN=inv(RA)  ;  %把逆提出来节省时间

function fenlei=fenlei(x)  %theta行向量 gamma列向量，x行向量
    gamma=gamma1(1:m);
    b=gamma1(m+1);
    f=gamma'*RAN*rA(theta0,x)+b;   
    if f>=0
        fenlei=1;
    else
        fenlei=-1;
    end
end
moxing=@fenlei;
node2=node;


end