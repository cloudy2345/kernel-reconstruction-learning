%�޻ع��� �������� logistic-loss
function moxing=xunlian4(x_train,y_train,x_node,y_node)   %���ط���ģ��
size1=size(x_train);
x_dim=size1(2);
n_train=size1(1);
size2=size(x_node);
m=size2(1);
lambda1=1;
theta0=0.4;       %30ά�е��
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


%---------------------------------theta-loss--------------------------------- 
function loss=loss(theta,gamma1)  %(gamma1Ϊgamma��b)��������theta������
    RA=RAC(theta);
    RAN=inv(RA)  ;  %�����������ʡʱ��
    loss=0;
    gamma=gamma1(1:m);
    for i=1:n_train
        bx=[RAN*rA(theta,x_train(i,:));1]  ;   %����Ϊ�лع�����ʽ
        f=gamma1'*bx;      %����ֵ
        loss=loss+log(1+exp(-2.*y_train(i,:).*f));
    end
    loss=loss+0.5.*lambda1.*gamma'*RAN*gamma;
end

%---------------------------------gamma-loss---------------------------------
function [los,lp,lh]=fprim(gamma1,theta)  %���� ���� hession
    RA=RAC(theta);
    RAN=inv(RA)  ;  %�����������ʡʱ��
    RANX=zeros(m+1);
    RANX(1:m,1:m)=RAN ;  %Ϊ�˹���ͬ�лع�����ʽ
    los=0;
    gamma=gamma1(1:m);
    RANX(1:m,1:m)=RAN;
    for i=1:n_train
        bx=[RAN*rA(theta,x_train(i,:));1]  ;   %����Ϊ�лع�����ʽ
        f=gamma1'*bx;      %����ֵ
        los=los+log(1+exp(-2.*y_train(i,:).*f));
    end
    los=los+0.5.*lambda1.*gamma'*RAN*gamma;
    
    if nargout > 1   % ����fun������Ҫ�����������������
    lp=0;
    for i=1:n_train
        bx=[RAN*rA(theta,x_train(i,:));1] ;
        f=gamma1'*bx;
        lp=lp-2.*y_train(i).*bx/(1+exp(2.*y_train(i).*f));
    end
    lp=lp+lambda1.*RANX*gamma1;
    end
    
    if nargout > 2   % ����fun������Ҫ�����������������
    lh=0;
    for i=1:n_train
        bx=[RAN*rA(theta,x_train(i,:));1] ;
        f=gamma1'*bx;
        lh=lh+4.*y_train(i).^2.*exp(2.*y_train(i).*f)./(1+exp(2.*y_train(i).*f)).^2.*bx*bx';
    end
    lh=lh+lambda1.*RANX   ; 
    lh=(lh+lh')./2;
    end
end

%---------------------------------���Ʋ���--------------------------------
cha=100;
options = optimset('Algorithm','trust-region-reflective' ,'GradObj','on', 'Hessian','on');
while cha>0.1 | cha<-1
    [gamma1,loss1,exitflag1]=fmincon(@(gamma) fprim(gamma,theta0),ones(m+1,1),[],[],[],[],[],[],[],options);  %gamma1����gamma��b
    loss1
    [theta0,loss2,exitflag2]=fmincon(@(theta) loss(theta,gamma1),1,[],[],[],[],0.000001,4); %ͬ��
    theta0
    loss2
    cha=loss1-loss2;
end      %�ҵ�theta0��gamma
%------------------------------�������---------------------------------
RA=RAC(theta0);
RAN=inv(RA)  ;  %�����������ʡʱ��
function fenlei=fenlei(x)  %theta������ gamma��������x������
    bx=[RAN*rA(theta0,x);1] ;
    f=gamma1'*bx;
    if f>=0
        fenlei=1;
    else
        fenlei=-1;
    end
end
moxing=@fenlei;

end