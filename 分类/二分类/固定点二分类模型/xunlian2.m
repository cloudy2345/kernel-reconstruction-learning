%�лع��� �������� logistic-loss
function moxing=xunlian2(x_train,y_train,x_node,y_node)   %���ط���ģ��
size1=size(x_train);
x_dim=size1(2);
n_train=size1(1);
size2=size(x_node);
m=size2(1);
lambda1=1;
theta0=1;       %30ά�е��
%---------------------------kernel--------------------------------
function k=kernel(theta,x,y)         %������ theta>0
    k=exp(-1/x_dim.*sum((x-y).^2.*theta));
end

%--------------------------------------����ʹ�ú���----------------------------------
function RA=RAC(theta)  %����RA
    RA=eye(m);   %�Խ�����1
    %ֻ�����ϰ벿��
    for i1=1:m-1
        for j1=i1+1:m
            RA(i1,j1)=kernel(theta,x_node(i1,:),x_node(j1,:));  
        end
    end
    RA=RA+RA'-eye(m)+10^(-6).*eye(m);
end
function r=rA(theta,x)   %ra(x)������������,x������
    r=zeros(m,1);
    for i2=1:m
        r(i2)=kernel(theta,x,x_node(i2,:));
    end
end
GA=[ones(m,1),x_node];

%---------------------------------theta-loss--------------------------------- 
function loss=loss(theta,gamma)  %(gamma)��������theta������
    RA=RAC(theta);
    RAN=inv(RA)  ;  %�����������ʡʱ��
    GAN=inv(GA'*RAN*GA);  %��ʡ��ʱ��
    U=RAN*GA*GAN;
    V=(eye(m)-RAN*GA*GAN*GA')*RAN;
    loss=0;
    for i3=1:n_train
        g=[1,x_train(i3,:)]';
        b=U*g+V*rA(theta,x_train(i3,:));
        f=gamma'*b;      %����ֵ
        loss=loss+log(1+exp(-2.*y_train(i3).*f));
    end
    loss=loss+0.5.*lambda1.*gamma'*V*RA*V'*gamma;      %%%%%%%%%%%
end

%---------------------------------gamma-loss---------------------------------
function [los,lp,lh]=fprim(gamma,theta)  %���� ���� hession
    RA=RAC(theta);
    RAN=inv(RA)  ;  %�����������ʡʱ��
    GAN=inv(GA'*RAN*GA);  %��ʡ��ʱ��
    U=RAN*GA*GAN;
    V=(eye(m)-RAN*GA*GAN*GA')*RAN;
    los=0;
    for i4=1:n_train
        g=[1,x_train(i4,:)]';
        b=U*g+V*rA(theta,x_train(i4,:));
        f=gamma'*b;      %����ֵ
        los=los+log(1+exp(-2.*y_train(i4).*f));
    end
    los=los+0.5.*lambda1.*gamma'*V*RA*V'*gamma;        %%%%%%%%%%%%
    
    if nargout > 1   % ����fun������Ҫ�����������������
    lp=0;
    for i5=1:n_train
        g=[1,x_train(i5,:)]';
        b=U*g+V*rA(theta,x_train(i5,:));
        f=gamma'*b;      %����ֵ
        lp=lp-2.*y_train(i5).*b/(1+exp(2.*y_train(i5).*f));
    end
    lp=lp+lambda1.*V*RA*V'*gamma;          %%%%%%%%%%%
    end
    
    if nargout > 2   % ����fun������Ҫ�����������������
    lh=0;
    for i6=1:n_train
        g=[1,x_train(i6,:)]';
        b=U*g+V*rA(theta,x_train(i6,:));
        f=gamma'*b;      %����ֵ
        lh=lh+4.*y_train(i6).^2.*exp(2.*y_train(i6).*f)./(1+exp(2.*y_train(i6).*f)).^2.*b*b';
    end
    lh=lh+lambda1.*V*RA*V'   ;                %%%%%%%%%%%%%
    lh=(lh+lh')./2;
    end
end
%---------------------------------���Ʋ���--------------------------------
cha=100;
options = optimset('Algorithm','trust-region-reflective' ,'GradObj','on', 'Hessian','on');
while cha>0.1 | cha<-1
    [gamma,loss1,exitflag1]=fmincon(@(gamma) fprim(gamma,theta0),ones(m,1),[],[],[],[],[],[],[],options);
    loss1
    [theta0,loss2,exitflag2]=fmincon(@(theta) loss(theta,gamma),theta0,[],[],[],[],0.00001,[]); %ͬ��
    theta0
    loss2
    cha=loss1-loss2;
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