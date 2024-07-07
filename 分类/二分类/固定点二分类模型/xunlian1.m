%�лع��� �������� hinge-loss
function moxing=xunlian1(x_train,y_train,x_node,y_node)   %���ط���ģ��
size1=size(x_train);
x_dim=size1(2);
n_train=size1(1);
size2=size(x_node);
m=size2(1);
lambda1=1;
theta0=1;

%---------------------------kernel--------------------------------
function k=kernel(theta,x,y)         %������ theta>0
   % k=exp(-1/x_dim.*sum((x-y).^2.*thetab.*theta));
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

%--------------------------------SQP--------------------------------------
function [canshu,loss,exitflag]=SQP(theta)  %ǰ��m��gamma��������t
    RA=RAC(theta);
    RAN=inv(RA)  ;  %�����������ʡʱ��
    GAN=inv(GA'*RAN*GA);  %��ʡ��ʱ��
    U=RAN*GA*GAN;
    V=(eye(m)-RAN*GA*GAN*GA')*RAN;
    H=zeros(m+n_train);
    e=V*RA*V';                     %%%%%%%%%%%%%V*RA*V'
    H(1:m,1:m)=(e+ e')/2   ;      %H
    f=ones(m+n_train,1);
    f(1:m)=0      ;           %f
    b=[zeros(n_train,1);-ones(n_train,1)] ;  %b
    A1=[zeros(n_train,m),-eye(n_train)];
    A2=zeros(n_train,m);
    for i=1:n_train
        g=[1,x_train(i,:)]';
        b1=U*g+V*rA(theta,x_train(i,:));
        A2(i,:)=-1.*y_train(i).*b1';
    end
    A2=[A2,-eye(n_train)];
    A=[A1;A2]     ;       %A
    [canshu,loss,exitflag]=quadprog(lambda1.*H,f,A,b);
end

%----------------------------------loss--------------------------------- 
function loss=loss(theta,gamma)  %(gamma)��������theta������
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
        loss=loss+max(1-y_train(i).*f,0);
    end
    loss=loss+0.5.*lambda1.*gamma'*V*RA*V'*gamma;     %%%%%%%%%%%%
end
%---------------------------------���Ʋ���--------------------------------
cha=100;
guo=0;
while cha>0.1 
    [canshu,loss1,exitflag1]=SQP(theta0) ;
    loss1
    gamma=canshu(1:m);
    [theta0,loss2,exitflag2]=fmincon(@(theta) loss(theta,gamma),theta0,[],[],[],[],0.0001,[]); %ͬ��
    loss2
    theta0
    cha=loss1-loss2;
    guo=guo+1;
end      %�ҵ�theta0��gamma

%------------------------------�������---------------------------------
RA=RAC(theta0);
RAN=inv(RA)  ;  %�����������ʡʱ��
GAN=inv(GA'*RAN*GA);  %��ʡ��ʱ��
U=RAN*GA*GAN;
V=(eye(m)-RAN*GA*GAN*GA')*RAN;
function fenlei=fenlei(x)  %theta������ gamma��������x������
    g=[1,x]';
    b=U*g+V*rA(theta0,x);
    f=gamma'*b;      %����ֵ
    if f>=0
        fenlei=1;
    else
        fenlei=-1;
    end
end
 moxing=@fenlei;
end