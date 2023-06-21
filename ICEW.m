function [T] = ICEW(X, lambda,mu,alpha, patchSize, slideStep,m,n)
tol = 1e-3; 
max_iter = 500;
rho = 1.05;
max_mu = 1e10; 
DEBUG = 1;
img = zeros(m,n);
N = rankN(X, 0.1); 
dim = size(X);
r = dim(3);
T = zeros(dim);
Y = zeros(dim);
S = zeros(dim);
weightTen = ones(dim);

for iter = 1 : max_iter   
    preT = sum(T(:) ~= 0);  
    % update B
    B = prox_pstnn(-S-T+X-Y/mu, N, mu);
%     update S
    Q = X-B-T-Y/mu;
    if iter==1
        for i = 1:r
            tmp = norm(Q(:,:,i), 'fro');
            if tmp > alpha/mu
                coef = (tmp - alpha/mu) / tmp;
                S(:,:, i) = coef * Q(:,:,i);
            end
        end
    else
        for i = 1:r
            tmp = norm(Q(:,:,i), 'fro');
            if tmp > alpha/mu /norm(lambda1W(:,:,i), 'fro')
                coef = (tmp - alpha/mu /norm(lambda1W(:,:,i), 'fro')) / tmp;
                S(:,:, i) = coef * Q(:,:,i);
            end
        end
    end  

    % update T    
    T = prox_l1_mine(-S-B+X-Y/mu, weightTen*lambda/mu); 
    % update weightTen 
    TRY = B+T;
    TRY_R = res_patch_ten_mean(TRY, img, patchSize, slideStep);
    
    [lambda1, lambda2] = structure_tensor_lambda(TRY_R);
    cornerStrength = (((lambda1.*lambda2)./(lambda1 + lambda2+0.001)));
    maxValue = (max(lambda1,lambda2)); 
    cornerStrength = cornerStrength.*maxValue;
    priorWeight = mat2gray(cornerStrength); %%Corner Strength
    tenW = gen_patch_ten(priorWeight, patchSize, slideStep);   
    lambda1s = mat2gray((lambda1 - lambda2)); %%Edge Strength
    lambda1W = gen_patch_ten(lambda1s, patchSize, slideStep);
    weightTen = N./ T./tenW;
    dY = B+T+S-X; 
    err = norm(dY(:))/norm(X(:)); 
    if DEBUG
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                   ', err=' num2str(err)...
                    ',|T|0 = ' num2str(sum(T(:) > 0)) ]);
    end
    currT = sum(T(:) ~= 0); 
    if err < tol || (preT>0 && currT>0 && preT == currT)
        Spositive = max(T,0);
        Snegtive = min(T,0);
        normP = norm(res_patch_ten_mean(Spositive.*tenW, img, patchSize, slideStep),1);
        normN = norm(res_patch_ten_mean(Snegtive.*tenW, img, patchSize, slideStep),1);
        if normP>normN
            T = Spositive;
        else
            T = -Snegtive;
        end
        break;
    end 
    Y = Y + dY*mu; 
    mu = min(rho*mu,max_mu); 
end

function N = rankN(X, ratioN)
    [~,~,n3] = size(X);
    D = Unfold(X,n3,1);
    [~, S, ~] = svd(D, 'econ'); 
    [desS, ~] = sort(diag(S), 'descend'); 
    ratioVec = desS / desS(1); 
    idxArr = find(ratioVec < ratioN);
    if idxArr(1) > 1
        N = idxArr(1) - 1; 
    else
        N = 1;
    end