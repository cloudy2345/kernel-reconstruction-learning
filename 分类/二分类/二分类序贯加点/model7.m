%有回归项 各向异性 logistic-loss   进行后续加点设计,根据p加
function node2=model7(x_train,y_train,x_node,y_node,node)   %返回分类模型
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
GA=[ones(m,1),x_node];

%---------------------------------theta-loss--------------------------------- 
function loss=los(theta,gamma)  %(gamma)列向量，theta行向量
    RA=RAC(theta);
    RAN=inv(RA)  ;  %把逆提出来节省时间
    GAN=inv(GA'*RAN*GA);  %节省点时间
    U=RAN*GA*GAN;
    V=(eye(m)-RAN*GA*GAN*GA')*RAN;
    loss=0;
    for i=1:n_train
        g=[1,x_train(i,:)]';
        b=U*g+V*rA(theta,x_train(i,:));
        f=gamma'*b;      %函数值
        loss=loss+log(1+exp(-2.*y_train(i).*f));
    end
    loss=loss+0.5.*lambda1.*gamma'*(V*RA*V'+10^(-8).*eye(m))*gamma;     %%%%%  (V*RA*V'+10^(-8).*eye(m))
end

%---------------------------------gamma-loss---------------------------------
function [los,lp,lh]=fprim(gamma,theta)  %本身 导数 hession
    RA=RAC(theta);
    RAN=inv(RA)  ;  %把逆提出来节省时间
    GAN=inv(GA'*RAN*GA);  %节省点时间
    U=RAN*GA*GAN;
    V=(eye(m)-RAN*GA*GAN*GA')*RAN;
    los=0;
    for i=1:n_train
        g=[1,x_train(i,:)]';
        b=U*g+V*rA(theta,x_train(i,:));
        f=gamma'*b;      %函数值
        los=los+log(1+exp(-2.*y_train(i).*f));
    end
    los=los+0.5.*lambda1.*gamma'*(V*RA*V'+10^(-8).*eye(m))*gamma;    %%%
    
    if nargout > 1   % 调用fun函数并要求有两个输出变量。
    lp=0;
    for i=1:n_train
        g=[1,x_train(i,:)]';
        b=U*g+V*rA(theta,x_train(i,:));
        f=gamma'*b;      %函数值
        lp=lp-2.*y_train(i).*b/(1+exp(2.*y_train(i).*f));
    end
    lp=lp+lambda1.*(V*RA*V'+10^(-8).*eye(m))*gamma;      %%%%
    end
    
    if nargout > 2   % 调用fun函数并要求有两个输出变量。
    lh=0;
    for i=1:n_train
        g=[1,x_train(i,:)]';
        b=U*g+V*rA(theta,x_train(i,:));
        f=gamma'*b;      %函数值
        lh=lh+4.*y_train(i).^2.*exp(2.*y_train(i).*f)./(1+exp(2.*y_train(i).*f)).^2.*b*b';
    end
    lh=lh+lambda1.*(V*RA*V'+10^(-8).*eye(m))  ;     %%%%
    lh=(lh+lh')./2;
    end
end



%---------------------------------估计参数--------------------------------
cha=100;
gamma0=y_node;
options = optimset('Algorithm','trust-region-reflective' ,'GradObj','on', 'Hessian','on');
while cha>0.1 | cha <-0.3
    [gamma0,loss1,exitflag1]=fmincon(@(gamma) fprim(gamma,theta0),gamma0,[],[],[],[],[],[],[],options);
    loss1
   % [theta0,loss2,exitflag2]=fmincon(@(theta) loss(theta,gamma),theta0,[],[],[],[],0.0001.*ones(1,x_dim),[]);
    [theta0,loss2,exitflag2]=fmincon(@(theta) los(theta,gamma0),theta0,[],[],[],[],0,[]); %同性
    loss2
    cha=loss1-loss2;
end      %找到theta0与gamma


%----------------------------------------开始新加点--------------------------------------------
RA=RAC(theta0);
RAN=inv(RA)  ;  %把逆提出来节省时间
GAN=inv(GA'*RAN*GA);  %节省点时间
U=RAN*GA*GAN;
V=(eye(m)-RAN*GA*GAN*GA')*RAN;

Zong=1:n_train;
S=setdiff(Zong,node);  %备选集

m=length(node);
x_node=x_train(node,:);
K_1=zeros(n_train,m);
for i13=1:n_train
    g=[1,x_train(i13,:)]';
    b=U*g+V*rA(theta0,x_train(i13,:));
    K_1(i13,:)=b';
end
K_2=RAN;                         %%%%%%%%
P=1./(1+exp(-1.*y_train.*(K_1*gamma0)));
P_yu=P(S);  %在剩余点的p

KK=0;


while true
    [min1,min_weizhi]=min(P_yu);  %最小值于最小值的位置 
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
    GA(m,:)=[1,x_train(node_jia,:)];          %把RA，GA等调整正确
    RAN=inv(RA)  ;  %把逆提出来节省时间
    GAN=inv(GA'*RAN*GA);  %节省点时间
    U=RAN*GA*GAN;
    V=(eye(m)-RAN*GA*GAN*GA')*RAN;
    K_1=zeros(n_train,m);
    for i13=1:n_train
        g=[1,x_train(i13,:)]';
        b=U*g+V*rA(theta0,x_train(i13,:));
        K_1(i13,:)=b';
    end
    
    [gamma0,loss1,exitflag1]=fmincon(@(gamma) fprim(gamma,theta0),y_node,[],[],[],[],[],[],[],options);%再次估计gamma
    P=1./(1+exp(-1.*y_train.*(K_1*gamma0)));
    S=setdiff(Zong,node);
    P_yu=P(S);  %在剩余点的p
    
    %if abs(H11-H_old)/H_old<0.001 | KK>20
    KK=KK+1
   % if KK>=5 | min1>0.9
   if KK>=jia
        break
   end
    
   if mod(KK,5)==0  %5步冲估计一次theta
        [theta0,loss2,exitflag2]=fmincon(@(theta) los(theta,gamma0),theta0,[],[],[],[],0,[]); %同性
   end
    
end

%------------------------------计算误差---------------------------------

x_node=x_train(node,:);
m=length(node);
RA=RAC(theta0);
RAN=inv(RA)  ;  %把逆提出来节省时间
GAN=inv(GA'*RAN*GA);  %节省点时间
U=RAN*GA*GAN;
V=(eye(m)-RAN*GA*GAN*GA')*RAN;
function fenlei=fenlei(x)  %theta行向量 gamma列向量，x行向量
    g=[1,x]';
    b=U*g+V*rA(theta0,x);
    f=gamma0'*b;      %函数值
    if f>=0
        fenlei=1;
    else
        fenlei=-1;
    end
end
moxing=@fenlei;
node2=node;


end
