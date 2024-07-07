%������logistic,����ͬ��
function moxing=xunlian3(x_train,y_train,x_node)   %���ط���ģ�� ,ȡx_node=x_train
size1=size(x_train);
x_dim=size1(2);
n_train=size1(1);
size2=size(x_node);
m=size2(1);
lambda1=1;
theta0=fmincon(@(theta) julia(x_train,y_train,theta),1,[],[]);
leibie=length(unique(y_train));
theta0=theta0./leibie;

Y=zeros(n_train,leibie);   %���Ƕ�������,��ʾ�����ĸ���
for i=1:leibie
    Y(y_train==i,i)=1;
end
%---------------------------kernel--------------------------------
function k=kernel(theta,x,y)         %������ theta>0
    k=exp(-1/x_dim.*sum((x-y).^2.*theta'));
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


%---------------------------------theta-loss--------------------------------- 
function loss=los1(theta,gamma)  %(gamma)��������theta������
    RA=RAC(theta);
    loss=0;
    gammaC=reshape(gamma,m,leibie);  %ÿ�ж�Ӧgammac������ȡ������
    VVRA=RA;
    for i1=1:leibie-1
        VVRA=blkdiag(VVRA,RA);
    end

    for i3=1:n_train
        b=RA(:,i3)  ;   
        f=gammaC'*b   ;      %����ֵ��������leibie�� 
        loss=loss+log(sum(exp(f)))-Y(i3,:)*f;
    end
    loss=loss+0.5.*lambda1.*(gamma'*VVRA*gamma);
end

%---------------------------------gamma-loss---------------------------------
function [los,lp,lh]=fprim(theta,gamma)  %���� ���� hession
    RA=RAC(theta);
    VVRA=RA;
    for i1=1:leibie-1
        VVRA=blkdiag(VVRA,RA);
    end
    gammaC=reshape(gamma,m,leibie);  %ÿ�ж�Ӧgammac������ȡ������
    los=0;
    for i4=1:n_train
        b=RA(:,i4) ;   %����Ϊ�޻ع�����ʽ
        f=gammaC'*b;      %����ֵ
        los=los+log(sum(exp(f)))-Y(i4,:)*f;
    end
    los=los+0.5.*lambda1.*(gamma'*VVRA*gamma);        %%%%%%%%%%%%
    
    if nargout > 1   % ����fun������Ҫ�����������������
    lp=zeros(m*leibie,1);
    for i5=1:n_train
        b=RA(:,i5)  ;   %����Ϊ�޻ع�����ʽ
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
        b=RA(:,i6)  ;   %����Ϊ�޻ع�����ʽ
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
gamma0=ones(m*leibie,1)./10;
while cha>15 | cha<-3
    [gamma0,loss1,exitflag1]=fmincon(@(gamma) fprim(theta0,gamma),gamma0,[],[],[],[],[],[],[],options);  %gamma1����gamma��b
    loss1
    [theta0,loss2,exitflag2]=fmincon(@(theta) los1(theta,gamma0),theta0,[],[],[],[],0,[]); %ͬ��
    theta0
    loss2
    cha=loss1-loss2;
end      %�ҵ�theta0��gamma
%------------------------------�������---------------------------------
RA=RAC(theta0);
gammaC=reshape(gamma0,m,leibie);  %ÿ�ж�Ӧgammac������ȡ������
function fen=fenlei(x)  %theta������ gamma��������x������
    b=rA(theta0,x) ;
    f=gammaC'*b;      %����ֵ
    [mm1,fen]=max(f);
end
moxing=@fenlei;


end