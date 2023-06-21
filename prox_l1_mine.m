function x = prox_l1_mine(b,lambda)

x = sign(b).*(max(0,abs(b)-abs(lambda)));
