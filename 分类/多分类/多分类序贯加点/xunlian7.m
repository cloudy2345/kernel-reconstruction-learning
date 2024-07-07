%�лع��� ����ͬ�� logistic-loss  ����� ���к����ӵ����,����p��
function node2=xunlian7(x_train,y_train,x_node,node)   %���ط���ģ��
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

Y=zeros(n_train,leibie);   %���Ƕ�������,��ʾ�����ĸ���
for i0=1:leibie
    Y(y_train==i0,i0)=1;
end

%---------------------------kernel--------------------------------
function k=kernel(theta,x,y)         %������ theta>0
    k=exp(-1/x_dim.*sum((x-y).^2.*theta'));
end

%--------------------------------------����ʹ�ú���----------------------------------
function RA=RAC(theta)  %����RA
    RA=eye(m);   %�Խ�����1
    %ֻ�����ϰ벿��
    for i=1:m-1
        for j=i+1:m
            RA(i,j)=kernel(theta,x_node(i,:),x_node(j,:));  
        end
    end
    RA=RA+RA'-eye(m)+10^(-6).*eye(m);
end
function r=rA(theta,x)   %ra(x)������������,x������
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
function loss=los1(theta,gamma)  %(gamma)��������theta������
    RA=RAC(theta);
    RAN=inv(RA)  ;  %�����������ʡʱ��
    GAN=inv(GA'*RAN*GA);  %��ʡ��ʱ��
    U=RAN*GA*GAN;
    V=(eye(m)-RAN*GA*GAN*GA')*RAN;
    VRV=V*RA*V'+10^(-8).*eye(m);
    VVRA=VRV;
    for i1=1:leibie-1
        VVRA=blkdiag(VVRA,VRV);
    end
    loss=0;
    gammaC=reshape(gamma,m,leibie);  %ÿ�ж�Ӧgammac������ȡ������
    for i3=1:n_train
        g=[1,x_train(i3,:)]';
        b=U*g+V*rA(theta,x_train(i3,:));
        f=gammaC'*b   ;      %����ֵ��������leibie��  
        loss=loss+log(sum(exp(f)))-Y(i3,:)*f;
    end
    loss=loss+0.5.*lambda1.*(gamma'*VVRA*gamma);      %%%%%%%%%%%
end

%---------------------------------gamma-loss---------------------------------
function [los,lp,lh]=fprim(theta,gamma)  %���� ���� hession
    RA=RAC(theta);
    RAN=inv(RA)  ;  %�����������ʡʱ��
    GAN=inv(GA'*RAN*GA);  %��ʡ��ʱ��
    U=RAN*GA*GAN;
    V=(eye(m)-RAN*GA*GAN*GA')*RAN;
    VRV=V*RA*V'+10^(-8).*eye(m);
    VVRA=VRV;
    for i1=1:leibie-1
        VVRA=blkdiag(VVRA,VRV);
    end
    gammaC=reshape(gamma,m,leibie);  %ÿ�ж�Ӧgammac������ȡ������
    los=0;
    for i4=1:n_train
        g=[1,x_train(i4,:)]';
        b=U*g+V*rA(theta,x_train(i4,:));
        f=gammaC'*b;      %����ֵ
        los=los+log(sum(exp(f)))-Y(i4,:)*f;
    end
    los=los+0.5.*lambda1.*(gamma'*VVRA*gamma);        %%%%%%%%%%%%
    
    if nargout > 1   % ����fun������Ҫ�����������������
    lp=zeros(m*leibie,1);
    for i5=1:n_train
        g=[1,x_train(i5,:)]';
        b=U*g+V*rA(theta,x_train(i5,:));
        f=gammaC'*b;      %����ֵ
        for k1=1:leibie
            lp(m*(k1-1)+1:m*k1)=lp(m*(k1-1)+1:m*k1)-b.*(y_train(i5)==k1)+ exp(f(k1)).*b./sum(exp(f));
        end
    end

    lp=lp+lambda1.*VVRA*gamma;
    end
    
    if nargout > 2   % ����fun������Ҫ�����������������
    lh=zeros(m*leibie,m*leibie);
    for i6=1:n_train
        g=[1,x_train(i6,:)]';
        b=U*g+V*rA(theta,x_train(i6,:));
        f=gammaC'*b;      %����ֵ
        for k1=1:leibie-1    %��һ��
            for k2=k1+1:leibie
                lh(m*(k1-1)+1:m*k1,m*(k2-1)+1:m*k2)=lh(m*(k1-1)+1:m*k1,m*(k2-1)+1:m*k2)-exp(f(k1)).*exp(f(k2)).*b*b'./(sum(exp(f))).^2;
            end
        end
        for k1=1:leibie    %�Խ���
            lh(m*(k1-1)+1:m*k1,m*(k1-1)+1:m*k1)=lh(m*(k1-1)+1:m*k1,m*(k1-1)+1:m*k1)+( exp(f(k1)).*sum(exp(f)).*b*b'-(exp(f(k1)))^2.*b*b' )./(sum(exp(f))).^2;
        end
    end
    lhduijiao=zeros(m*leibie,m*leibie);
    for k3=1:leibie
        lhduijiao((k3-1)*m+1:k3*m,(k3-1)*m+1:k3*m)=lh((k3-1)*m+1:k3*m,(k3-1)*m+1:k3*m);
    end
    lh=lh-lhduijiao; %������
    lh=lh+lh'+lhduijiao;
    lh=lh+lambda1.*VVRA;

    end
end

%---------------------------------���Ʋ���--------------------------------
cha=100;
options = optimset('Algorithm','trust-region-reflective' ,'GradObj','on', 'Hessian','on');
%options=optimoptions(@fmincon,'MaxFunEvals',10000);
gamma0=ones(m*leibie,1);
while cha>20 | cha< -5
 %   [gamma0,loss1,exitflag1]=fmincon(@(gamma) los1(theta0,gamma),gamma0,[],[],[],[],[],[],[],options);
    [gamma0,loss1,exitflag1]=fmincon(@(gamma) fprim(theta0,gamma),ones(m*leibie,1),[],[],[],[],[],[],[],options);
    loss1
    [theta0,loss2,exitflag2]=fmincon(@(theta) los1(theta,gamma0),theta0,[],[],[],[],0,[]); %ͬ��
    theta0
    loss2
    cha=loss1-loss2;
end      %�ҵ�theta0��gamma


%----------------------------------------��ʼ�¼ӵ�--------------------------------------------
RA=RAC(theta0);
RAN=inv(RA)  ;  %�����������ʡʱ��
GAN=inv(GA'*RAN*GA);  %��ʡ��ʱ��
U=RAN*GA*GAN;
V=(eye(m)-RAN*GA*GAN*GA')*RAN;

Zong=1:n_train;
S=setdiff(Zong,node);  %��ѡ��

m=length(node);
x_node=x_train(node,:);
K_1=zeros(n_train,m);  %ÿ��bxi
for i13=1:n_train
    g=[1,x_train(i13,:)]';
    b=U*g+V*rA(theta0,x_train(i13,:));
    K_1(i13,:)=b';
end
gammaC=reshape(gamma0,m,leibie);  %ÿ�ж�Ӧgammac������ȡ������
F=K_1*gammaC  ;  %ÿ����f(xi)'
P=zeros(n_train,1);
for i1=1:n_train
    P(i1)=Y(i1,:)*F(i1,:)' ;
end
P_yu=P(S);  %��ʣ����p

KK=0;

loss_old=loss2;
while true
    [min1,min_weizhi]=min(P_yu);  %��Сֵ����Сֵ��λ�� 
    node_jia=S(min_weizhi);
    m=m+1;
    node(m)=node_jia;        %��ӽ������µ�
    x_node(m,:)=x_train(node_jia,:);

    for i17=1:m-1
        RA(i17,m)=kernel(theta0,x_train(node_jia,:),x_node(i17));
        RA(m,i17)=RA(i17,m);
    end
    RA(m,m)=1;    %�µ�RA
    GA(m,:)=[1,x_train(node_jia,:)];          %��RA��GA�ȵ�����ȷ
    RAN=inv(RA)  ;  %�����������ʡʱ��
    GAN=inv(GA'*RAN*GA);  %��ʡ��ʱ��
    U=RAN*GA*GAN;
    V=(eye(m)-RAN*GA*GAN*GA')*RAN;
    K_1=zeros(n_train,m);  %ÿ��bxi
    for i13=1:n_train
        g=[1,x_train(i13,:)]';
        b=U*g+V*rA(theta0,x_train(i13,:));
        K_1(i13,:)=b';
    end
    
    [gamma0,loss1,exitflag1]=fmincon(@(gamma) fprim(theta0,gamma),ones(m*leibie,1),[],[],[],[],[],[],[],options);%�ٴι���gamma
    gammaC=reshape(gamma0,m,leibie);  %ÿ�ж�Ӧgammac������ȡ������
    F=K_1*gammaC  ;  %ÿ����f(xi)'
    P=zeros(n_train,1);
    for i14=1:n_train
        P(i14)=Y(i14,:)*F(i14,:)' ;
    end
    S=setdiff(Zong,node);
    P_yu=P(S);  %��ʣ����p
    
    loss_new=loss1;
    KK=KK+1
    if  KK>=jia
        break
    end
    loss_old=loss_new;
    if mod(KK,5)==0  %5�������һ��theta
        [theta0,loss2,exitflag2]=fmincon(@(theta) los1(theta,gamma0),theta0,[],[],[],[],0,[]); %ͬ��
    end
    
    
end

%------------------------------�������---------------------------------

x_node=x_train(node,:);
m=length(node);
RA=RAC(theta0);
RAN=inv(RA)  ;  %�����������ʡʱ��
GAN=inv(GA'*RAN*GA);  %��ʡ��ʱ��
U=RAN*GA*GAN;
V=(eye(m)-RAN*GA*GAN*GA')*RAN;
gammaC=reshape(gamma0,m,leibie);  %ÿ�ж�Ӧgammac������ȡ������
function fen=fenlei(x)  %theta������ gamma��������x������
    g=[1,x]';
    b=U*g+V*rA(theta0,x);
    f=gammaC'*b;      %����ֵ
    [mm1,fen]=max(f);
end
moxing=@fenlei;
node2=node;


end