%有回归项 各向异性 logistic-loss
function moxing=model2(x_train,y_train,x_node,y_node)   %返回分类模型
size1=size(x_train);
x_dim=size1(2);
n_train=size1(1);
size2=size(x_node);
m=size2(1);
lambda1=1;
theta0=1;       %30维有点大
%---------------------------kernel--------------------------------
function k=kernel(theta,x,y)         %行向量 theta>0
    k=exp(-1/x_dim.*sum((x-y).^2.*theta));
end

%--------------------------------------构建使用函数----------------------------------
function RA=RAC(theta)  %产生RA
    RA=eye(m);   %对角线是1
    %只算了上半部分
    for i1=1:m-1
        for j1=i1+1:m
            RA(i1,j1)=kernel(theta,x_node(i1,:),x_node(j1,:));  
        end
    end
    RA=RA+RA'-eye(m)+10^(-6).*eye(m);
end
function r=rA(theta,x)   %ra(x)到结点的列向量,x行向量
    r=zeros(m,1);
    for i2=1:m
        r(i2)=kernel(theta,x,x_node(i2,:));
    end
end
GA=[ones(m,1),x_node];

%---------------------------------theta-loss--------------------------------- 
function loss=loss(theta,gamma)  %(gamma)列向量，theta行向量
    RA=RAC(theta);
    RAN=inv(RA)  ;  %把逆提出来节省时间
    GAN=inv(GA'*RAN*GA);  %节省点时间
    U=RAN*GA*GAN;
    V=(eye(m)-RAN*GA*GAN*GA')*RAN;
    loss=0;
    for i3=1:n_train
        g=[1,x_train(i3,:)]';
        b=U*g+V*rA(theta,x_train(i3,:));
        f=gamma'*b;      %函数值
        loss=loss+log(1+exp(-2.*y_train(i3).*f));
    end
    loss=loss+0.5.*lambda1.*gamma'*V*RA*V'*gamma;      %%%%%%%%%%%
end

%---------------------------------gamma-loss---------------------------------
function [los,lp,lh]=fprim(gamma,theta)  %本身 导数 hession
    RA=RAC(theta);
    RAN=inv(RA)  ;  %把逆提出来节省时间
    GAN=inv(GA'*RAN*GA);  %节省点时间
    U=RAN*GA*GAN;
    V=(eye(m)-RAN*GA*GAN*GA')*RAN;
    los=0;
    for i4=1:n_train
        g=[1,x_train(i4,:)]';
        b=U*g+V*rA(theta,x_train(i4,:));
        f=gamma'*b;      %函数值
        los=los+log(1+exp(-2.*y_train(i4).*f));
    end
    los=los+0.5.*lambda1.*gamma'*V*RA*V'*gamma;        %%%%%%%%%%%%
    
    if nargout > 1   % 调用fun函数并要求有两个输出变量。
    lp=0;
    for i5=1:n_train
        g=[1,x_train(i5,:)]';
        b=U*g+V*rA(theta,x_train(i5,:));
        f=gamma'*b;      %函数值
        lp=lp-2.*y_train(i5).*b/(1+exp(2.*y_train(i5).*f));
    end
    lp=lp+lambda1.*V*RA*V'*gamma;          %%%%%%%%%%%
    end
    
    if nargout > 2   % 调用fun函数并要求有两个输出变量。
    lh=0;
    for i6=1:n_train
        g=[1,x_train(i6,:)]';
        b=U*g+V*rA(theta,x_train(i6,:));
        f=gamma'*b;      %函数值
        lh=lh+4.*y_train(i6).^2.*exp(2.*y_train(i6).*f)./(1+exp(2.*y_train(i6).*f)).^2.*b*b';
    end
    lh=lh+lambda1.*V*RA*V'   ;                %%%%%%%%%%%%%
    lh=(lh+lh')./2;
    end
end
%---------------------------------估计参数--------------------------------
cha=100;
options = optimset('Algorithm','trust-region-reflective' ,'GradObj','on', 'Hessian','on');
while cha>0.1 | cha<-1
    [gamma,loss1,exitflag1]=fmincon(@(gamma) fprim(gamma,theta0),ones(m,1),[],[],[],[],[],[],[],options);
    loss1
    [theta0,loss2,exitflag2]=fmincon(@(theta) loss(theta,gamma),theta0,[],[],[],[],0.00001,[]); %同性
    theta0
    loss2
    cha=loss1-loss2;
end      %找到theta0与gamma

%------------------------------计算误差---------------------------------

RA=RAC(theta0);
RAN=inv(RA)  ;  %把逆提出来节省时间
GAN=inv(GA'*RAN*GA);  %节省点时间
U=RAN*GA*GAN;
V=(eye(m)-RAN*GA*GAN*GA')*RAN;
function fenlei=fenlei(x)  %theta行向量 gamma列向量，x行向量
    g=[1,x]';
    b=U*g+V*rA(theta0,x);
    f=gamma'*b;      %函数值
    if f>=0
       fenlei=1;
    else
       fenlei=-1;
    end
end
moxing=@fenlei;



end
