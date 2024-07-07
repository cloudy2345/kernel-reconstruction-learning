function S=julia(x_train,y_train,theta)
size1=size(x_train);
x_dim=size1(2);
n_train=size1(1);
function k=kernel(theta,x,y)         %列向量 theta>0
    k=exp(-1/x_dim.*sum((x-y).^2.*theta'));
end
A=eye(n_train)  ;     %gram矩阵
for i=1:n_train-1
    for j=i+1:n_train
        A(i,j)=kernel(theta,x_train(i,:),x_train(j,:));
    end
end
A=A+A'-eye(n_train);

B=eye(n_train)  ;  %理想矩阵
for i=1:n_train-1
    for j=i+1:n_train
        if y_train(i)==y_train(j)
            B(i,j)=1;
        end
    end
end
B=B+B'-eye(n_train);
S=sum(sum((A-B).^2));

end