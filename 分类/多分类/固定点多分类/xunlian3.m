%多分类核logistic,各项同性
function moxing=xunlian3(x_train,y_train,x_node)   %返回分类模型 ,取x_node=x_train
size1=size(x_train);
x_dim=size1(2);
n_train=size1(1);
size2=size(x_node);
m=size2(1);
lambda1=1;
theta0=fmincon(@(theta) julia(x_train,y_train,theta),1,[],[]);
leibie=length(unique(y_train));
theta0=theta0./leibie;

Y=zeros(n_train,leibie);   %考虑多分类情况,表示属于哪个类
for i=1:leibie
    Y(y_train==i,i)=1;
end
%---------------------------kernel--------------------------------
function k=kernel(theta,x,y)         %列向量 theta>0
    k=exp(-1/x_dim.*sum((x-y).^2.*theta'));
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


%---------------------------------theta-loss--------------------------------- 
function loss=los1(theta,gamma)  %(gamma)列向量，theta列向量
    RA=RAC(theta);
    loss=0;
    gammaC=reshape(gamma,m,leibie);  %每列对应gammac，按列取排数据
    VVRA=RA;
    for i1=1:leibie-1
        VVRA=blkdiag(VVRA,RA);
    end

    for i3=1:n_train
        b=RA(:,i3)  ;   
        f=gammaC'*b   ;      %函数值，向量，leibie行 
        loss=loss+log(sum(exp(f)))-Y(i3,:)*f;
    end
    loss=loss+0.5.*lambda1.*(gamma'*VVRA*gamma);
end

%---------------------------------gamma-loss---------------------------------
function [los,lp,lh]=fprim(theta,gamma)  %本身 导数 hession
    RA=RAC(theta);
    VVRA=RA;
    for i1=1:leibie-1
        VVRA=blkdiag(VVRA,RA);
    end
    gammaC=reshape(gamma,m,leibie);  %每列对应gammac，按列取排数据
    los=0;
    for i4=1:n_train
        b=RA(:,i4) ;   %构建为无回归项形式
        f=gammaC'*b;      %函数值
        los=los+log(sum(exp(f)))-Y(i4,:)*f;
    end
    los=los+0.5.*lambda1.*(gamma'*VVRA*gamma);        %%%%%%%%%%%%
    
    if nargout > 1   % 调用fun函数并要求有两个输出变量。
    lp=zeros(m*leibie,1);
    for i5=1:n_train
        b=RA(:,i5)  ;   %构建为无回归项形式
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
        b=RA(:,i6)  ;   %构建为无回归项形式
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
gamma0=ones(m*leibie,1)./10;
while cha>15 | cha<-3
    [gamma0,loss1,exitflag1]=fmincon(@(gamma) fprim(theta0,gamma),gamma0,[],[],[],[],[],[],[],options);  %gamma1包含gamma与b
    loss1
    [theta0,loss2,exitflag2]=fmincon(@(theta) los1(theta,gamma0),theta0,[],[],[],[],0,[]); %同性
    theta0
    loss2
    cha=loss1-loss2;
end      %找到theta0与gamma
%------------------------------计算误差---------------------------------
RA=RAC(theta0);
gammaC=reshape(gamma0,m,leibie);  %每列对应gammac，按列取排数据
function fen=fenlei(x)  %theta行向量 gamma列向量，x行向量
    b=rA(theta0,x) ;
    f=gammaC'*b;      %函数值
    [mm1,fen]=max(f);
end
moxing=@fenlei;


end