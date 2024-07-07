function [x_train,x_test,x_node,fz]=shuju5()  %�ܶȹ���
n_train=600;  % ��100��
n_test=10000;
nsuiji=5000;
m=8;
x_dim=1;
alpha=0.4;
%----------------------------------����ѵ����--------------------------------------------
u1=-2 ;
u2=1;
v1=0.5;
v2=0.5;  %��׼��

x_train=zeros(n_train,x_dim);   %��ϸ�˹�ֲ�
for i=1:n_train
    if rand(1)<alpha
        x_train(i,:)=normrnd ( u1, v1,1 );
    else 
        x_train(i,:)=normrnd ( u2, v2, 1 );
    end
end


%----------------------�������Լ���mcmc������������������������������������������
x_test=zeros(n_test,x_dim);   %��ϸ�˹�ֲ�
for i=1:n_test
    if rand(1)<alpha
        x_test(i,:)=normrnd ( u1, v1,1 );
    else 
        x_test(i,:)=normrnd ( u2, v2, 1 );
    end
end

%----------------------------�ҽڵ�---------------------------------------
function zuida=juli(X)   %һ��m����ļ��ϵ��ھ���
    zuida=0;
    for i=1:m-1
        for j=i+1:m
            a=sum(1./abs(X(i,:)-X(j,:)));
            if a>zuida
                zuida=a;
            end
        end
    end
end
function minnode=jiedian(train)  %��x_train���ҵ���õ�m����
    zuid=10^10;
    for i=1:nsuiji
        a=randperm(n_train/2);
        weizhi=a(1:m);
        Y=juli(train(weizhi,:));
        if Y<zuid
            zuid=Y;
            minnode=weizhi;
        end
    end
end 
node=jiedian(x_train);
x_node=x_train(node,:);
%x_node=[3,-5,-1,0,-3]';


fz=@(x) alpha.*(2.*pi).^(-x_dim/2).*(det(v1.^2))^(-0.5)*exp(-(x'-u1)'*inv(v1.^2)*(x'-u1)./2)+(1-alpha).*(2.*pi).^(-x_dim/2).*(det(v2.^2))^(-0.5)*exp(-(x'-u2)'*inv(v2.^2)*(x'-u2)./2);

end