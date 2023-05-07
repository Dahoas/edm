import torch 

def create_matrix(n,k):
    A = torch.zeros(k,n)
    for i in range(n):
        A[i,i] = 1.
    for i in range(k-n):
        A[n+i,-2] = -i-1
        A[n+i,-1] = i+2
    return A
torch.set_printoptions(sci_mode=False,precision=3,linewidth=200)

n = 8
m = n//2 + 1
new_n = 2*n
new_m = new_n//2 + 1 

B = torch.arange(1,n*n+1).reshape((n,n))*1.
x_ft = torch.fft.rfft2(B).real
B = x_ft
print(B.shape)
A = create_matrix(n, new_n)
print(B)
AT = create_matrix(m, new_m).T
print(A@B@AT)