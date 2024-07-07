%有回归项 各向同性 logistic-loss  多分类 进行后续加点设计,根据p加
function node2=xunlian7(x_train,y_train,x_node,node)   %返回分类模型
size1=size(x_train);
x_dim=size1(2);
n_train=size1(1);
size2=size(x_node);
m=size2(1);
lambda1=1;
theta0=fmincon(@(theta) julia(x_train,y_train,theta),1,[],[]);
leibie=length(unique(y_train));
theta0=theta0./leibie;
jia=3*x_dim;

Y=zeros(n_train,leibie);   %考虑多分类情况,表示属于哪个类
for i0=1:leibie
    Y(y_train==i0,i0)=1;
end

%---------------------------kernel--------------------------------
function k=kernel(theta,x,y)         %列向量 theta>0
    k=exp(-1/x_dim.*sum((x-y).^2.*theta'));
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
function loss=los1(theta,gamma)  %(gamma)列向量，theta列向量
    RA=RAC(theta);
    RAN=inv(RA)  ;  %把逆提出来节省时间
    GAN=inv(GA'*RAN*GA);  %节省点时间
    U=RAN*GA*GAN;
    V=(eye(m)-RAN*GA*GAN*GA')*RAN;
    VRV=V*RA*V'+10^(-8).*eye(m);
    VVRA=VRV;
    for i1=1:leibie-1
        VVRA=blkdiag(VVRA,VRV);
    end
    loss=0;
    gammaC=reshape(gamma,m,leibie);  %每列对应gammac，按列取排数据
    for i3=1:n_train
        g=[1,x_train(i3,:)]';
        b=U*g+V*rA(theta,x_train(i3,:));
        f=gammaC'*b   ;      %函数值，向量，leibie行  
        loss=loss+log(sum(exp(f)))-Y(i3,:)*f;
    end
    loss=loss+0.5.*lambda1.*(gamma'*VVRA*gamma);      %%%%%%%%%%%
end

%---------------------------------gamma-loss---------------------------------
function [los,lp,lh]=fprim(theta,gamma)  %本身 导数 hession
    RA=RAC(theta);
    RAN=inv(RA)  ;  %把逆提出来节省时间
    GAN=inv(GA'*RAN*GA);  %节省点时间
    U=RAN*GA*GAN;
    V=(eye(m)-RAN*GA*GAN*GA')*RAN;
    VRV=V*RA*V'+10^(-8).*eye(m);
    VVRA=VRV;
    for i1=1:leibie-1
        VVRA=blkdiag(VVRA,VRV);
    end
    gammaC=reshape(gamma,m,leibie);  %每列对应gammac，按列取排数据
    los=0;
    for i4=1:n_train
        g=[1,x_train(i4,:)]';
        b=U*g+V*rA(theta,x_train(i4,:));
        f=gammaC'*b;      %函数值
        los=los+log(sum(exp(f)))-Y(i4,:)*f;
    end
    los=los+0.5.*lambda1.*(gamma'*VVRA*gamma);        %%%%%%%%%%%%
    
    if nargout > 1   % 调用fun函数并要求有两个输出变量。
    lp=zeros(m*leibie,1);
    for i5=1:n_train
        g=[1,x_train(i5,:)]';
        b=U*g+V*rA(theta,x_train(i5,:));
        f=gammaC'*b;      %函数值
        for k1=1:leibie
            lp(m*(k1-1)+1:m*k1)=lp(m*(k1-1)+1:m*k1)-b.*(y_train(i5)==k1)+ exp(f(k1)).*b./sum(exp(f));
        end
    end

    lp=lp+lambda1.*VVRA*gamma;
    end
    
    if nargout > 2   % 调用fun函数并要求有两个输出变量。
    lh=zeros(m*leibie,m*leibie);
    for i6=1:n_train
        g=[1,x_train(i6,:)]';
        b=U*g+V*rA(theta,x_train(i6,:));
        f=gammaC'*b;      %函数值
        for k1=1:leibie-1    %上一半
            for k2=k1+1:leibie
                lh(m*(k1-1)+1:m*k1,m*(k2-1)+1:m*k2)=lh(m*(k1-1)+1:m*k1,m*(k2-1)+1:m*k2)-exp(f(k1)).*exp(f(k2)).*b*b'./(sum(exp(f))).^2;
            end
        end
        for k1=1:leibie    %对角线
            lh(m*(k1-1)+1:m*k1,m*(k1-1)+1:m*k1)=lh(m*(k1-1)+1:m*k1,m*(k1-1)+1:m*k1)+( exp(f(k1)).*sum(exp(f)).*b*b'-(exp(f(k1)))^2.*b*b' )./(sum(exp(f))).^2;
        end
    end
    lhduijiao=zeros(m*leibie,m*leibie);
    for k3=1:leibie
        lhduijiao((k3-1)*m+1:k3*m,(k3-1)*m+1:k3*m)=lh((k3-1)*m+1:k3*m,(k3-1)*m+1:k3*m);
    end
    lh=lh-lhduijiao; %上三角
    lh=lh+lh'+lhduijiao;
    lh=lh+lambda1.*VVRA;

    end
end

%---------------------------------估计参数--------------------------------
cha=100;
options = optimset('Algorithm','trust-region-reflective' ,'GradObj','on', 'Hessian','on');
%options=optimoptions(@fmincon,'MaxFunEvals',10000);
gamma0=ones(m*leibie,1);
while cha>20 | cha< -5
 %   [gamma0,loss1,exitflag1]=fmincon(@(gamma) los1(theta0,gamma),gamma0,[],[],[],[],[],[],[],options);
    [gamma0,loss1,exitflag1]=fmincon(@(gamma) fprim(theta0,gamma),ones(m*leibie,1),[],[],[],[],[],[],[],options);
    loss1
    [theta0,loss2,exitflag2]=fmincon(@(theta) los1(theta,gamma0),theta0,[],[],[],[],0,[]); %同性
    theta0
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
K_1=zeros(n_train,m);  %每行bxi
for i13=1:n_train
    g=[1,x_train(i13,:)]';
    b=U*g+V*rA(theta0,x_train(i13,:));
    K_1(i13,:)=b';
end
gammaC=reshape(gamma0,m,leibie);  %每列对应gammac，按列取排数据
F=K_1*gammaC  ;  %每行是f(xi)'
P=zeros(n_train,1);
for i1=1:n_train
    P(i1)=Y(i1,:)*F(i1,:)' ;
end
P_yu=P(S);  %在剩余点的p

KK=0;

loss_old=loss2;
while true
    [min1,min_weizhi]=min(P_yu);  %最小值于最小值的位置 
    node_jia=S(min_weizhi);
    m=m+1;
    node(m)=node_jia;        %添加进来最新的
    x_node(m,:)=x_train(node_jia,:);

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
    K_1=zeros(n_train,m);  %每行bxi
    for i13=1:n_train
        g=[1,x_train(i13,:)]';
        b=U*g+V*rA(theta0,x_train(i13,:));
        K_1(i13,:)=b';
    end
    
    [gamma0,loss1,exitflag1]=fmincon(@(gamma) fprim(theta0,gamma),ones(m*leibie,1),[],[],[],[],[],[],[],options);%再次估计gamma
    gammaC=reshape(gamma0,m,leibie);  %每列对应gammac，按列取排数据
    F=K_1*gammaC  ;  %每行是f(xi)'
    P=zeros(n_train,1);
    for i14=1:n_train
        P(i14)=Y(i14,:)*F(i14,:)' ;
    end
    S=setdiff(Zong,node);
    P_yu=P(S);  %在剩余点的p
    
    loss_new=loss1;
    KK=KK+1
    if  KK>=jia
        break
    end
    loss_old=loss_new;
    if mod(KK,5)==0  %5步冲估计一次theta
        [theta0,loss2,exitflag2]=fmincon(@(theta) los1(theta,gamma0),theta0,[],[],[],[],0,[]); %同性
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
gammaC=reshape(gamma0,m,leibie);  %每列对应gammac，按列取排数据
function fen=fenlei(x)  %theta行向量 gamma列向量，x行向量
    g=[1,x]';
    b=U*g+V*rA(theta0,x);
    f=gammaC'*b;      %函数值
    [mm1,fen]=max(f);
end
moxing=@fenlei;
node2=node;


end