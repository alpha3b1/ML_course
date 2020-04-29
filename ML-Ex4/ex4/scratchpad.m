%load thetas
load('ex4weights.mat');

y_ = zeros(m,10);
for row = 1:size(y_,1)
    y_(row,y(row)) =1;
end

X = [ones(m, 1) X];
a_2 = sigmoid(X*Theta1');
a_2 = [ones(m,1) a_2];
a_3 = sigmoid(a_2*Theta2');

tic
tot_all_rows = 0;

for p=1:m
    for q=1:10
        tot_all_rows = tot_all_rows +(-y_(p,q)*log(a_3(p,:))(q) - (1-y_(p,q))*log(1-a_3(p,:))(q));
    end
end
toc
tot_all_rows

tic
tot_all_rows = 0;

for p=1:m
   tot_all_rows = tot_all_rows + (-y_(p,:)*log(a_3(p,:))'  -(1-y_(p,:))*log(1-a_3(p,:))') ;
end
toc
tot_all_rows/m

%-y_(1,:)*log(a_3(1,:))'  -(1-y_(1,:))*log(1-a_3(1,:))'

respuesta correcta: 0.383770
valor sin regularizacion: 0.287629
