%�лع��� �������� logistic-loss   ���к����ӵ����,����p��
function node2=xunlian7(x_train,y_train,x_node,y_node,node)   %���ط���ģ��
size1=size(x_train);
x_dim=size1(2);
n_train=size1(1);
size2=size(x_node);
m=size2(1);
lambda1=0.3;
theta0=0.4;
jia=3*x_dim;

%---------------------------kernel--------------------------------
function k=kernel(theta,x,y)         %������ theta>0
    k=exp(-1/x_dim.*sum((x-y).^2.*theta));
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
function loss=los(theta,gamma)  %(gamma)��������theta������
    RA=RAC(theta);
    RAN=inv(RA)  ;  %�����������ʡʱ��
    GAN=inv(GA'*RAN*GA);  %��ʡ��ʱ��
    U=RAN*GA*GAN;
    V=(eye(m)-RAN*GA*GAN*GA')*RAN;
    loss=0;
    for i=1:n_train
        g=[1,x_train(i,:)]';
        b=U*g+V*rA(theta,x_train(i,:));
        f=gamma'*b;      %����ֵ
        loss=loss+log(1+exp(-2.*y_train(i).*f));
    end
    loss=loss+0.5.*lambda1.*gamma'*(V*RA*V'+10^(-8).*eye(m))*gamma;     %%%%%  (V*RA*V'+10^(-8).*eye(m))
end

%---------------------------------gamma-loss---------------------------------
function [los,lp,lh]=fprim(gamma,theta)  %���� ���� hession
    RA=RAC(theta);
    RAN=inv(RA)  ;  %�����������ʡʱ��
    GAN=inv(GA'*RAN*GA);  %��ʡ��ʱ��
    U=RAN*GA*GAN;
    V=(eye(m)-RAN*GA*GAN*GA')*RAN;
    los=0;
    for i=1:n_train
        g=[1,x_train(i,:)]';
        b=U*g+V*rA(theta,x_train(i,:));
        f=gamma'*b;      %����ֵ
        los=los+log(1+exp(-2.*y_train(i).*f));
    end
    los=los+0.5.*lambda1.*gamma'*(V*RA*V'+10^(-8).*eye(m))*gamma;    %%%
    
    if nargout > 1   % ����fun������Ҫ�����������������
    lp=0;
    for i=1:n_train
        g=[1,x_train(i,:)]';
        b=U*g+V*rA(theta,x_train(i,:));
        f=gamma'*b;      %����ֵ
        lp=lp-2.*y_train(i).*b/(1+exp(2.*y_train(i).*f));
    end
    lp=lp+lambda1.*(V*RA*V'+10^(-8).*eye(m))*gamma;      %%%%
    end
    
    if nargout > 2   % ����fun������Ҫ�����������������
    lh=0;
    for i=1:n_train
        g=[1,x_train(i,:)]';
        b=U*g+V*rA(theta,x_train(i,:));
        f=gamma'*b;      %����ֵ
        lh=lh+4.*y_train(i).^2.*exp(2.*y_train(i).*f)./(1+exp(2.*y_train(i).*f)).^2.*b*b';
    end
    lh=lh+lambda1.*(V*RA*V'+10^(-8).*eye(m))  ;     %%%%
    lh=(lh+lh')./2;
    end
end



%---------------------------------���Ʋ���--------------------------------
cha=100;
gamma0=y_node;
options = optimset('Algorithm','trust-region-reflective' ,'GradObj','on', 'Hessian','on');
while cha>0.1 | cha <-0.3
    [gamma0,loss1,exitflag1]=fmincon(@(gamma) fprim(gamma,theta0),gamma0,[],[],[],[],[],[],[],options);
    loss1
   % [theta0,loss2,exitflag2]=fmincon(@(theta) loss(theta,gamma),theta0,[],[],[],[],0.0001.*ones(1,x_dim),[]);
    [theta0,loss2,exitflag2]=fmincon(@(theta) los(theta,gamma0),theta0,[],[],[],[],0,[]); %ͬ��
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
K_1=zeros(n_train,m);
for i13=1:n_train
    g=[1,x_train(i13,:)]';
    b=U*g+V*rA(theta0,x_train(i13,:));
    K_1(i13,:)=b';
end
K_2=RAN;                         %%%%%%%%
P=1./(1+exp(-1.*y_train.*(K_1*gamma0)));
P_yu=P(S);  %��ʣ����p

KK=0;


while true
    [min1,min_weizhi]=min(P_yu);  %��Сֵ����Сֵ��λ�� 
    node_jia=S(min_weizhi);
    m=m+1;
    node(m)=node_jia;        %��ӽ������µ�
    x_node(m,:)=x_train(node_jia,:);
    y_node(m)=y_train(node_jia);
    
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
    K_1=zeros(n_train,m);
    for i13=1:n_train
        g=[1,x_train(i13,:)]';
        b=U*g+V*rA(theta0,x_train(i13,:));
        K_1(i13,:)=b';
    end
    
    [gamma0,loss1,exitflag1]=fmincon(@(gamma) fprim(gamma,theta0),y_node,[],[],[],[],[],[],[],options);%�ٴι���gamma
    P=1./(1+exp(-1.*y_train.*(K_1*gamma0)));
    S=setdiff(Zong,node);
    P_yu=P(S);  %��ʣ����p
    
    %if abs(H11-H_old)/H_old<0.001 | KK>20
    KK=KK+1
   % if KK>=5 | min1>0.9
   if KK>=jia
        break
   end
    
   if mod(KK,5)==0  %5�������һ��theta
        [theta0,loss2,exitflag2]=fmincon(@(theta) los(theta,gamma0),theta0,[],[],[],[],0,[]); %ͬ��
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
function fenlei=fenlei(x)  %theta������ gamma��������x������
    g=[1,x]';
    b=U*g+V*rA(theta0,x);
    f=gamma0'*b;      %����ֵ
    if f>=0
        fenlei=1;
    else
        fenlei=-1;
    end
end
moxing=@fenlei;
node2=node;


end