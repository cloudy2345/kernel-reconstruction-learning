function moxing=density5(x_train,x_node,x_test)   %返回密度模型，没有回归项,加上一个距离基函数,quasimonte
size1=size(x_train);
x_dim=size1(2);
n_train=size1(1);
size2=size(x_node);
n_test=size(x_test,1);
m=size2(1);
p = haltonset(x_dim,'Skip',1e3,'Leap',1e2);
nn1=5e6; %
xx1=p(1:nn1,:);   %loss中积分用
mmin=(min(x_train)-2);
mmax=(max(x_train)+2);   %积分上下界
Pro=prod(mmax-mmin);
xx1=(mmax-mmin).*xx1+mmin ;

nn2=1e7; %   最后用
xx2=p(1:nn2,:);  
xx2=(mmax-mmin).*xx2+mmin ;
    
%---------------------------kernel--------------------------------
function k=kernel(theta,x,y)         %列向量 theta>0
    k=exp(-1.*sum((x-y).^2.*theta));
end
%-----------------------------------theta的选择---------------------------
davg=0;
for i1=1:m-1
    for j1=i1+1:m
        davg=davg+1/sum((x_node(i1,:)-x_node(j1,:)).^2);
    end
end
davg=(davg*2/m/(m-1))^(-0.5);
theta0=log(2)/davg^2./3;   %roshan说要一个小于他一个大于他，但是其实不好用
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
function r_xa=r_xa(theta)   %n_train行m列，R_XA
    r_xa=zeros(n_train,m);
    for i=1:n_train
        r_xa(i,:)=rA(theta,x_train(i,:))';
    end
end
function r_xxa=r_xxa(theta)   %nn行m列，R_XXA
    r_xxa=zeros(nn1,m);
    for i=1:nn1
        r_xxa(i,:)=rA(theta,xx1(i,:))';
    end
end

%----------------------------------loss--------------------------------- 
function C=CC3(theta,gamma,xx)         %积分的值,theta自己，,nn是随机数个数,quasi Monte carlo
    RA=RAC(theta);
    RAN=inv(RA)  ;  %把逆提出来节省时间
    nn=length(xx);
    RXXXA=zeros(nn,m);
    for i9=1:nn
        RXXXA(i9,:)=rA(theta,xx(i9,:))';
    end
    C=mean(exp(RXXXA*RAN*gamma));
    C=C*Pro;
end
         
function [loss,lp,lh]=fprim(theta,gamma)  %前面theta，后面gamma
    loss=-mean(RXARAN0*gamma);
    exprxxarangamma=exp(RXXARAN0*gamma);
    meanexprxxarangamma=mean(exprxxarangamma);   %多次使用
    clear exprxxarangamma
    loss=loss+log(Pro.*meanexprxxarangamma);
    
    if nargout > 1   % 调用fun函数并要求有两个输出变量。
    lp=-(mean(RXARAN0,1))';
    meanxiangliang=(mean(RXXARAN0.*exp(RXXARAN0*gamma),1))';  %列向量
    lp=lp+1./meanexprxxarangamma.*meanxiangliang;
    end
    
    if nargout > 2   % 调用fun函数并要求有两个输出变量。
    lh=1./meanexprxxarangamma.* (RXXARAN0'*diag(exprxxarangamma)*RXXARAN0)./nn1;
    lh=lh-1./meanexprxxarangamma.^2.* meanxiangliang*meanxiangliang';
    lh=(lh+lh')./2;
    end
end

function loss=los(theta,gamma)   %优化theta
    RA=RAC(theta);
    RAN=inv(RA)  ;  %把逆提出来节省时间
    RXA=r_xa(theta);
    RXXA=r_xxa(theta);
    loss=-mean(RXA*RAN*gamma);
    loss=loss+log(Pro.*mean(exp(RXXA*RAN*gamma)));   
end
%---------------------------------估计参数--------------------------------
tic
%[beta0,loss3,exitflag3]=fmincon(@los2,[theta0;-0.1;zeros(m,1)],[-1,1,zeros(1,m)],0);
cha=100;
gamma0=0.1.*ones(m,1);
kkk=1;  %至少两轮
%theta0固定
RA0=RAC(theta0);
RAN0=inv(RA0)  ;  %把逆提出来节省时间
RXA0=r_xa(theta0);   %n_train*m
RXXA0=r_xxa(theta0);  %nn1*m
RXARAN0=RXA0*RAN0;  %多次用，与gamma无关
RXXARAN0=RXXA0*RAN0;

while cha>90 | cha<-90   % | kkk<2         %--------------------------------------
  %  options=optimoptions(@fmincon,'MaxFunEvals',100000);
  %  options =optimset('Algorithm','trust-region-reflective','LargeScale','on','GradObj','on','Hessian','on','Maxfunevals',100000); 
    options =optimset('GradObj','on');
    [gamma0,loss1,exitflag1]=fmincon(@(gamma) fprim(theta0,gamma),gamma0, [], [],[],[],[],[],[],options );   %第一项是alpha
    loss1
%    [theta0,loss2,exitflag2]=fmincon(@(theta) los(theta,gamma0),theta0 ,[],[],[],[],0.00001,[] );
 %   loss2
    cha=1;
    kkk=kkk+1;
end      %找到theta0与gamma
%------------------------------计算误差---------------------------------

clear RXXA0  %内存不足
clear RXARAN0
clear RXXARAN0
clear RXA0
toc

C=CC3(theta0,gamma0,xx2); 
RA1=RAC(theta0);
RAN1=inv(RA1)  ;  %把逆提出来节省时间
Rt=zeros(n_test,m);
for i99=1:n_test
    Rt(i99,:)=rA(theta0,x_test(i99,:))';
end
fg=exp(Rt*RAN1*gamma0)./C;

moxing=fg;


end