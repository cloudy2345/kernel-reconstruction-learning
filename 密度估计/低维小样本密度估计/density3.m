function moxing=density3(x_train,x_node)   %返回密度模型，没有回归项,加上一个距离基函数
size1=size(x_train);
x_dim=size1(2);
n_train=size1(1);
size2=size(x_node);
m=size2(1);

%---------------------------kernel--------------------------------
function k=kernel(theta,x,y)         %列向量 theta>0
    k=exp(-1.*sum((x-y).^2.*theta'));
end
%-----------------------------------theta的选择---------------------------
davg=0;
for i1=1:m-1
    for j1=i1+1:m
        davg=davg+1/sum((x_node(i1,:)-x_node(j1,:)).^2);
    end
end
davg=(davg*2/m/(m-1))^(-0.5);
theta0=log(2)/davg^2./3;  %roshan说要一个小于他一个大于他，但是其实不好用
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
function rA_Z=rA_ZC(theta)   %m行n列，每一列是一个b(xi)
    rA_Z=zeros(m,n_train);
    for i=1:n_train
        rA_Z(:,i)=rA(theta,x_train(i,:));
    end
end

%----------------------------------loss--------------------------------- 
function C=CC(theta,beta)         %积分的值,theta自己，beta的第一项是alpha<0,第二项是gamma
    alpha=beta(1);
    gamma=beta(2:m+1);
    RA=RAC(theta);
    RAN=inv(RA)  ;  %把逆提出来节省时间
    h=RAN*gamma;
    if x_dim==1
        Gx=@(x1) (0);
        for i=1:m
            Gx=@(x1) ( h(i).*exp( -theta.* (x_node(i)-x1).^2 )+ alpha.*(x1-x_node(i)).^2 + Gx(x1)  );
        end
        Gx=@(x1) (exp(Gx(x1)));
        C=integral(Gx,min(x_train)-2,2+max(x_train));
    elseif x_dim==2
        Gx=@(x1,x2) (0);
        for i=1:m
            Gx=@(x1,x2) ( h(i).*exp( -theta.*( (x_node(i,1)-x1).^2+ (x_node(i,2)-x2).^2 ) )+ alpha.*((x_node(i,1)-x1).^2+ (x_node(i,2)-x2).^2) +Gx(x1,x2) );
        end
        Gx=@(x1,x2) (exp(Gx(x1,x2)));
        %Gx=@(x1,x2) exp( arrayfun(@(A) dot(h,rA(theta,[A,x2])), x1)) ;
        C=integral2(Gx,min(x_train(:,1))-2,max(x_train(:,1))+2,min(x_train(:,2))-2,max(x_train(:,2))+2);
    elseif x_dim==3
        Gx=@(x1,x2,x3) (0);
        for i=1:m
            Gx=@(x1,x2,x3) ( h(i).*exp( -theta.*( (x_node(i,1)-x1).^2+ (x_node(i,2)-x2).^2+(x_node(i,3)-x3).^2 ) )+ alpha.*((x_node(i,1)-x1).^2+ (x_node(i,2)-x2).^2+(x_node(i,3)-x3).^2 ) +Gx(x1,x2,x3) );
        end
        Gx=@(x1,x2,x3) (exp(Gx(x1,x2,x3)));
        C=integral3(Gx,min(x_train(:,1))-2,2+max(x_train(:,1)),min(x_train(:,2))-2,2+max(x_train(:,2)),min(x_train(:,3))-2,2+max(x_train(:,3)));
    end
end
function C=CC2(theta,beta,nn)         %积分的值，网格求和，nn^d是样本量
    alpha=beta(1);
    gamma=beta(2:m+1);
    RA=RAC(theta);
    RAN=inv(RA)  ;  %把逆提出来节省时间
    h=RAN*gamma;
    AA=zeros(nn,x_dim) ;  %网格取为100个，每一列是该维度的间隔点
    for i1=1:x_dim
        AA(:,i1)=linspace(min(x_train(:,i1))-2, max(x_train(:,i1))+2 ,nn)';    %这里影响后面的积分范围
    end
    BB=zeros(nn^x_dim,x_dim);  %存储取的值
    if x_dim==1
        BB=AA;
        k=nn;
    elseif x_dim==2
        [B1,B11]=meshgrid(AA(:,1)',AA(:,2)');  %B1与B11是100*100，B1的ij元结合B11的ij元得到格子点ij
        k=0;
        for i1=1:nn
            for j1=1:nn
                k=k+1;
                BB(k,:)=[B1(i1,j1),B11(i1,j1)];
            end
        end
    elseif x_dim==3
        [B2,B22,B222]=meshgrid(AA(:,1)',AA(:,2)',AA(:,3));  %B2与B22是100*100*100，B2的ijk元结合B22的ijk加上B222元得到格子点ijk
        k=0;
        for i1=1:nn
            for j1=1:nn
                for k1=1:nn
                    k=k+1;
                    BB(k,:)=[B2(i1,j1,k1),B22(i1,j1,k1),B222(i1,j1,k1)];
                end
            end
        end
    end
    C=0;
    for i3=1:k
        C=C+exp(h'*rA(theta,BB(i3,:))+ alpha.*sum(sum((x_node-BB(i3,:)).^2))  ); 
    end
    if x_dim==1
        C=C*(range(x_train)+4)/(nn-1);
    elseif x_dim==2
        C=C*(range(x_train(:,1))+4)*(range(x_train(:,2))+4)/(nn-1)^2;
    elseif x_dim==3
        C=C*(range(x_train(:,1))+4)*(range(x_train(:,2))+4)*(range(x_train(:,3))+4)/(nn-1)^3;
    end
end

function C=CC3(theta,beta,nn)         %积分的值,theta自己，beta的第一项是alpha<0,第二项是gamma,nn是随机数个数,quasi Monte carlo
    alpha=beta(1);
    gamma=beta(2:m+1);
    RA=RAC(theta);
    RAN=inv(RA)  ;  %把逆提出来节省时间
    h=RAN*gamma;
    p = haltonset(x_dim,'Skip',1e3,'Leap',1e2);    
    xx=p(1:nn,:);  %序列，n*d      这个其实可以拿出去，因为每次都一样
    mmin=(min(x_train)-2);
    mmax=(max(x_train)+2);   %积分上下界
    ff=@(x) exp(h'*rA(theta,x)+alpha.*sum(sum((x_node-x).^2)))  ;  %x是行向量
    C=0;
    for i=1:nn
        C=C+ff( (mmax-mmin).*xx(i,:)+mmin );
    end
    C=C*prod(mmax-mmin)/nn;
end
         


function los=los1(theta,beta)  %前面theta，后面gamma,gamma第一项其实是alpha<0
    alpha=beta(1);
    gamma=beta(2:m+1);
    RA=RAC(theta);
    RAN=inv(RA)  ;  %把逆提出来节省时间
    C=CC3(theta,beta,2000);                            %%%%%%%%%%%%%%%-------------------------------
    los=0;
    rA_Z=rA_ZC(theta);
    for i=1:n_train
        los=los+alpha.*sum(sum((x_node-x_train(i,:)).^2));  %距离的平方
    end
    los=(-sum(rA_Z'*RAN*gamma)-los)/n_train+log(C);
end

function los=los2(beta)  %前面theta，后面gamma,一快优化
    theta=beta(1);
    alpha=beta(2);
    gamma=beta(3:m+2);
    RA=RAC(theta);
    RAN=inv(RA)  ;  %把逆提出来节省时间
    C=CC(theta,[alpha;gamma]);
    los=0;
    rA_Z=rA_ZC(theta);
    for i=1:n_train
        los=los+alpha.*sum(sum((x_node-x_train(i,:)).^2));  %距离的平方
    end
    los=(-sum(rA_Z'*RAN*gamma)-los)/n_train+log(C);
end


%---------------------------------估计参数--------------------------------
%[beta0,loss3,exitflag3]=fmincon(@los2,[theta0;-0.1;zeros(m,1)],[-1,1,zeros(1,m)],0);
cha=100;
beta0=[-0.1;0.1.*ones(m,1)];
kkk=1;  %至少两轮
while cha>1.5 | cha<-1.5   % | kkk<2
   % options=optimoptions(@fmincon,'MaxFunEvals',10000);
    [beta0,loss1,exitflag1]=fmincon(@(beta) los1(theta0,beta),beta0, [1,zeros(1,m)], 0 );   %第一项是alpha
    loss1
    [theta0,loss2,exitflag2]=fmincon(@(theta) los1(theta,beta0),theta0,[],[],[],[],0,[]);
    loss2
    cha=loss1-loss2;
    kkk=kkk+1;
end      %找到theta0与gamma
beta0=[theta0;beta0];
%------------------------------计算误差---------------------------------

theta0 =beta0(1);
alpha0=beta0(2);
gamma0=beta0(3:m+2);
C=CC3(theta0,[alpha0;gamma0],10000);    %%%%%%%%%%%%-------------------------------------------
RA=RAC(theta0);
RAN=inv(RA)  ;  %把逆提出来节省时间
h=RAN*gamma0;
function md=midu(x)  %theta列向量 gamma列向量，x行向量
    md=exp(h'*rA(theta0,x)+alpha0.*sum(sum( (x_node-x).^2) ))./C;
end
moxing=@midu


end