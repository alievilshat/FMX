from numbapro import cuda
from candidate import getcandidate
from polynom import p
import numpy as np

BLOKS = 1
THREADS = 1

@cuda.autojit
def calculate(d_p, d_c, d_r):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= d_c.size:
        return

    stat = False
    for j in range(128):
        if (        (1 == d_p[d_c[i,0],0] * (j & 64) + d_p[d_c[i,1],0] * (j & 32) + d_p[d_c[i,2],0] * (j & 16) + d_p[d_c[i,3],0] * (j & 8) + d_p[d_c[i,4],0] * (j & 4) + d_p[d_c[i,5],0] * (j & 2) + d_p[d_c[i,6],0] * (j & 1))           #ae
                and (0 == d_p[d_c[i,0],1] * (j & 64) + d_p[d_c[i,1],1] * (j & 32) + d_p[d_c[i,2],1] * (j & 16) + d_p[d_c[i,3],1] * (j & 8) + d_p[d_c[i,4],1] * (j & 4) + d_p[d_c[i,5],1] * (j & 2) + d_p[d_c[i,6],1] * (j & 1))           #af
                and (0 == d_p[d_c[i,0],2] * (j & 64) + d_p[d_c[i,1],2] * (j & 32) + d_p[d_c[i,2],2] * (j & 16) + d_p[d_c[i,3],2] * (j & 8) + d_p[d_c[i,4],2] * (j & 4) + d_p[d_c[i,5],2] * (j & 2) + d_p[d_c[i,6],2] * (j & 1))           #ag
                and (0 == d_p[d_c[i,0],3] * (j & 64) + d_p[d_c[i,1],3] * (j & 32) + d_p[d_c[i,2],3] * (j & 16) + d_p[d_c[i,3],3] * (j & 8) + d_p[d_c[i,4],3] * (j & 4) + d_p[d_c[i,5],3] * (j & 2) + d_p[d_c[i,6],3] * (j & 1))           #ah
                and (0 == d_p[d_c[i,0],4] * (j & 64) + d_p[d_c[i,1],4] * (j & 32) + d_p[d_c[i,2],4] * (j & 16) + d_p[d_c[i,3],4] * (j & 8) + d_p[d_c[i,4],4] * (j & 4) + d_p[d_c[i,5],4] * (j & 2) + d_p[d_c[i,6],4] * (j & 1))           #be
                and (0 == d_p[d_c[i,0],5] * (j & 64) + d_p[d_c[i,1],5] * (j & 32) + d_p[d_c[i,2],5] * (j & 16) + d_p[d_c[i,3],5] * (j & 8) + d_p[d_c[i,4],5] * (j & 4) + d_p[d_c[i,5],5] * (j & 2) + d_p[d_c[i,6],5] * (j & 1))           #bf
                and (1 == d_p[d_c[i,0],6] * (j & 64) + d_p[d_c[i,1],6] * (j & 32) + d_p[d_c[i,2],6] * (j & 16) + d_p[d_c[i,3],6] * (j & 8) + d_p[d_c[i,4],6] * (j & 4) + d_p[d_c[i,5],6] * (j & 2) + d_p[d_c[i,6],6] * (j & 1))           #bg
                and (0 == d_p[d_c[i,0],7] * (j & 64) + d_p[d_c[i,1],7] * (j & 32) + d_p[d_c[i,2],7] * (j & 16) + d_p[d_c[i,3],7] * (j & 8) + d_p[d_c[i,4],7] * (j & 4) + d_p[d_c[i,5],7] * (j & 2) + d_p[d_c[i,6],7] * (j & 1))           #bh
                and (0 == d_p[d_c[i,0],8] * (j & 64) + d_p[d_c[i,1],8] * (j & 32) + d_p[d_c[i,2],8] * (j & 16) + d_p[d_c[i,3],8] * (j & 8) + d_p[d_c[i,4],8] * (j & 4) + d_p[d_c[i,5],8] * (j & 2) + d_p[d_c[i,6],8] * (j & 1))           #ce
                and (0 == d_p[d_c[i,0],9] * (j & 64) + d_p[d_c[i,1],9] * (j & 32) + d_p[d_c[i,2],9] * (j & 16) + d_p[d_c[i,3],9] * (j & 8) + d_p[d_c[i,4],9] * (j & 4) + d_p[d_c[i,5],9] * (j & 2) + d_p[d_c[i,6],9] * (j & 1))           #cf
                and (0 == d_p[d_c[i,0],10] * (j & 64) + d_p[d_c[i,1],10] * (j & 32) + d_p[d_c[i,2],10] * (j & 16) + d_p[d_c[i,3],10] * (j & 8) + d_p[d_c[i,4],10] * (j & 4) + d_p[d_c[i,5],10] * (j & 2) + d_p[d_c[i,6],10] * (j & 1))            #cg
                and (0 == d_p[d_c[i,0],11] * (j & 64) + d_p[d_c[i,1],11] * (j & 32) + d_p[d_c[i,2],11] * (j & 16) + d_p[d_c[i,3],11] * (j & 8) + d_p[d_c[i,4],11] * (j & 4) + d_p[d_c[i,5],11] * (j & 2) + d_p[d_c[i,6],11] * (j & 1))            #ch
                and (0 == d_p[d_c[i,0],12] * (j & 64) + d_p[d_c[i,1],12] * (j & 32) + d_p[d_c[i,2],12] * (j & 16) + d_p[d_c[i,3],12] * (j & 8) + d_p[d_c[i,4],12] * (j & 4) + d_p[d_c[i,5],12] * (j & 2) + d_p[d_c[i,6],12] * (j & 1))            #de
                and (0 == d_p[d_c[i,0],13] * (j & 64) + d_p[d_c[i,1],13] * (j & 32) + d_p[d_c[i,2],13] * (j & 16) + d_p[d_c[i,3],13] * (j & 8) + d_p[d_c[i,4],13] * (j & 4) + d_p[d_c[i,5],13] * (j & 2) + d_p[d_c[i,6],13] * (j & 1))            #df
                and (0 == d_p[d_c[i,0],14] * (j & 64) + d_p[d_c[i,1],14] * (j & 32) + d_p[d_c[i,2],14] * (j & 16) + d_p[d_c[i,3],14] * (j & 8) + d_p[d_c[i,4],14] * (j & 4) + d_p[d_c[i,5],14] * (j & 2) + d_p[d_c[i,6],14] * (j & 1))            #dg
                and (0 == d_p[d_c[i,0],15] * (j & 64) + d_p[d_c[i,1],15] * (j & 32) + d_p[d_c[i,2],15] * (j & 16) + d_p[d_c[i,3],15] * (j & 8) + d_p[d_c[i,4],15] * (j & 4) + d_p[d_c[i,5],15] * (j & 2) + d_p[d_c[i,6],15] * (j & 1))):          #dh
            stat = True
    
    if stat:
        stat = False
        for j in range(128):
            if (        (0 == d_p[d_c[i,0],0] * (j & 64) + d_p[d_c[i,1],0] * (j & 32) + d_p[d_c[i,2],0] * (j & 16) + d_p[d_c[i,3],0] * (j & 8) + d_p[d_c[i,4],0] * (j & 4) + d_p[d_c[i,5],0] * (j & 2) + d_p[d_c[i,6],0] * (j & 1))            #ae
                    and (1 == d_p[d_c[i,0],1] * (j & 64) + d_p[d_c[i,1],1] * (j & 32) + d_p[d_c[i,2],1] * (j & 16) + d_p[d_c[i,3],1] * (j & 8) + d_p[d_c[i,4],1] * (j & 4) + d_p[d_c[i,5],1] * (j & 2) + d_p[d_c[i,6],1] * (j & 1))            #af
                    and (0 == d_p[d_c[i,0],2] * (j & 64) + d_p[d_c[i,1],2] * (j & 32) + d_p[d_c[i,2],2] * (j & 16) + d_p[d_c[i,3],2] * (j & 8) + d_p[d_c[i,4],2] * (j & 4) + d_p[d_c[i,5],2] * (j & 2) + d_p[d_c[i,6],2] * (j & 1))            #ag
                    and (0 == d_p[d_c[i,0],3] * (j & 64) + d_p[d_c[i,1],3] * (j & 32) + d_p[d_c[i,2],3] * (j & 16) + d_p[d_c[i,3],3] * (j & 8) + d_p[d_c[i,4],3] * (j & 4) + d_p[d_c[i,5],3] * (j & 2) + d_p[d_c[i,6],3] * (j & 1))            #ah
                    and (0 == d_p[d_c[i,0],4] * (j & 64) + d_p[d_c[i,1],4] * (j & 32) + d_p[d_c[i,2],4] * (j & 16) + d_p[d_c[i,3],4] * (j & 8) + d_p[d_c[i,4],4] * (j & 4) + d_p[d_c[i,5],4] * (j & 2) + d_p[d_c[i,6],4] * (j & 1))            #be
                    and (0 == d_p[d_c[i,0],5] * (j & 64) + d_p[d_c[i,1],5] * (j & 32) + d_p[d_c[i,2],5] * (j & 16) + d_p[d_c[i,3],5] * (j & 8) + d_p[d_c[i,4],5] * (j & 4) + d_p[d_c[i,5],5] * (j & 2) + d_p[d_c[i,6],5] * (j & 1))            #bf
                    and (0 == d_p[d_c[i,0],6] * (j & 64) + d_p[d_c[i,1],6] * (j & 32) + d_p[d_c[i,2],6] * (j & 16) + d_p[d_c[i,3],6] * (j & 8) + d_p[d_c[i,4],6] * (j & 4) + d_p[d_c[i,5],6] * (j & 2) + d_p[d_c[i,6],6] * (j & 1))            #bg
                    and (1 == d_p[d_c[i,0],7] * (j & 64) + d_p[d_c[i,1],7] * (j & 32) + d_p[d_c[i,2],7] * (j & 16) + d_p[d_c[i,3],7] * (j & 8) + d_p[d_c[i,4],7] * (j & 4) + d_p[d_c[i,5],7] * (j & 2) + d_p[d_c[i,6],7] * (j & 1))            #bh
                    and (0 == d_p[d_c[i,0],8] * (j & 64) + d_p[d_c[i,1],8] * (j & 32) + d_p[d_c[i,2],8] * (j & 16) + d_p[d_c[i,3],8] * (j & 8) + d_p[d_c[i,4],8] * (j & 4) + d_p[d_c[i,5],8] * (j & 2) + d_p[d_c[i,6],8] * (j & 1))            #ce
                    and (0 == d_p[d_c[i,0],9] * (j & 64) + d_p[d_c[i,1],9] * (j & 32) + d_p[d_c[i,2],9] * (j & 16) + d_p[d_c[i,3],9] * (j & 8) + d_p[d_c[i,4],9] * (j & 4) + d_p[d_c[i,5],9] * (j & 2) + d_p[d_c[i,6],9] * (j & 1))            #cf
                    and (0 == d_p[d_c[i,0],10] * (j & 64) + d_p[d_c[i,1],10] * (j & 32) + d_p[d_c[i,2],10] * (j & 16) + d_p[d_c[i,3],10] * (j & 8) + d_p[d_c[i,4],10] * (j & 4) + d_p[d_c[i,5],10] * (j & 2) + d_p[d_c[i,6],10] * (j & 1))         #cg
                    and (0 == d_p[d_c[i,0],11] * (j & 64) + d_p[d_c[i,1],11] * (j & 32) + d_p[d_c[i,2],11] * (j & 16) + d_p[d_c[i,3],11] * (j & 8) + d_p[d_c[i,4],11] * (j & 4) + d_p[d_c[i,5],11] * (j & 2) + d_p[d_c[i,6],11] * (j & 1))         #ch
                    and (0 == d_p[d_c[i,0],12] * (j & 64) + d_p[d_c[i,1],12] * (j & 32) + d_p[d_c[i,2],12] * (j & 16) + d_p[d_c[i,3],12] * (j & 8) + d_p[d_c[i,4],12] * (j & 4) + d_p[d_c[i,5],12] * (j & 2) + d_p[d_c[i,6],12] * (j & 1))         #de
                    and (0 == d_p[d_c[i,0],13] * (j & 64) + d_p[d_c[i,1],13] * (j & 32) + d_p[d_c[i,2],13] * (j & 16) + d_p[d_c[i,3],13] * (j & 8) + d_p[d_c[i,4],13] * (j & 4) + d_p[d_c[i,5],13] * (j & 2) + d_p[d_c[i,6],13] * (j & 1))         #df
                    and (0 == d_p[d_c[i,0],14] * (j & 64) + d_p[d_c[i,1],14] * (j & 32) + d_p[d_c[i,2],14] * (j & 16) + d_p[d_c[i,3],14] * (j & 8) + d_p[d_c[i,4],14] * (j & 4) + d_p[d_c[i,5],14] * (j & 2) + d_p[d_c[i,6],14] * (j & 1))         #dg
                    and (0 == d_p[d_c[i,0],15] * (j & 64) + d_p[d_c[i,1],15] * (j & 32) + d_p[d_c[i,2],15] * (j & 16) + d_p[d_c[i,3],15] * (j & 8) + d_p[d_c[i,4],15] * (j & 4) + d_p[d_c[i,5],15] * (j & 2) + d_p[d_c[i,6],15] * (j & 1))):       #dh
                stat = True
    if stat:
        stat = False        
        for j in range(128):
            if (        (0 == d_p[d_c[i,0],0] * (j & 64) + d_p[d_c[i,1],0] * (j & 32) + d_p[d_c[i,2],0] * (j & 16) + d_p[d_c[i,3],0] * (j & 8) + d_p[d_c[i,4],0] * (j & 4) + d_p[d_c[i,5],0] * (j & 2) + d_p[d_c[i,6],0] * (j & 1))            #ae
                    and (0 == d_p[d_c[i,0],1] * (j & 64) + d_p[d_c[i,1],1] * (j & 32) + d_p[d_c[i,2],1] * (j & 16) + d_p[d_c[i,3],1] * (j & 8) + d_p[d_c[i,4],1] * (j & 4) + d_p[d_c[i,5],1] * (j & 2) + d_p[d_c[i,6],1] * (j & 1))            #af
                    and (0 == d_p[d_c[i,0],2] * (j & 64) + d_p[d_c[i,1],2] * (j & 32) + d_p[d_c[i,2],2] * (j & 16) + d_p[d_c[i,3],2] * (j & 8) + d_p[d_c[i,4],2] * (j & 4) + d_p[d_c[i,5],2] * (j & 2) + d_p[d_c[i,6],2] * (j & 1))            #ag
                    and (0 == d_p[d_c[i,0],3] * (j & 64) + d_p[d_c[i,1],3] * (j & 32) + d_p[d_c[i,2],3] * (j & 16) + d_p[d_c[i,3],3] * (j & 8) + d_p[d_c[i,4],3] * (j & 4) + d_p[d_c[i,5],3] * (j & 2) + d_p[d_c[i,6],3] * (j & 1))            #ah
                    and (0 == d_p[d_c[i,0],4] * (j & 64) + d_p[d_c[i,1],4] * (j & 32) + d_p[d_c[i,2],4] * (j & 16) + d_p[d_c[i,3],4] * (j & 8) + d_p[d_c[i,4],4] * (j & 4) + d_p[d_c[i,5],4] * (j & 2) + d_p[d_c[i,6],4] * (j & 1))            #be
                    and (0 == d_p[d_c[i,0],5] * (j & 64) + d_p[d_c[i,1],5] * (j & 32) + d_p[d_c[i,2],5] * (j & 16) + d_p[d_c[i,3],5] * (j & 8) + d_p[d_c[i,4],5] * (j & 4) + d_p[d_c[i,5],5] * (j & 2) + d_p[d_c[i,6],5] * (j & 1))            #bf
                    and (0 == d_p[d_c[i,0],6] * (j & 64) + d_p[d_c[i,1],6] * (j & 32) + d_p[d_c[i,2],6] * (j & 16) + d_p[d_c[i,3],6] * (j & 8) + d_p[d_c[i,4],6] * (j & 4) + d_p[d_c[i,5],6] * (j & 2) + d_p[d_c[i,6],6] * (j & 1))            #bg
                    and (0 == d_p[d_c[i,0],7] * (j & 64) + d_p[d_c[i,1],7] * (j & 32) + d_p[d_c[i,2],7] * (j & 16) + d_p[d_c[i,3],7] * (j & 8) + d_p[d_c[i,4],7] * (j & 4) + d_p[d_c[i,5],7] * (j & 2) + d_p[d_c[i,6],7] * (j & 1))            #bh
                    and (1 == d_p[d_c[i,0],8] * (j & 64) + d_p[d_c[i,1],8] * (j & 32) + d_p[d_c[i,2],8] * (j & 16) + d_p[d_c[i,3],8] * (j & 8) + d_p[d_c[i,4],8] * (j & 4) + d_p[d_c[i,5],8] * (j & 2) + d_p[d_c[i,6],8] * (j & 1))            #ce
                    and (0 == d_p[d_c[i,0],9] * (j & 64) + d_p[d_c[i,1],9] * (j & 32) + d_p[d_c[i,2],9] * (j & 16) + d_p[d_c[i,3],9] * (j & 8) + d_p[d_c[i,4],9] * (j & 4) + d_p[d_c[i,5],9] * (j & 2) + d_p[d_c[i,6],9] * (j & 1))            #cf
                    and (0 == d_p[d_c[i,0],10] * (j & 64) + d_p[d_c[i,1],10] * (j & 32) + d_p[d_c[i,2],10] * (j & 16) + d_p[d_c[i,3],10] * (j & 8) + d_p[d_c[i,4],10] * (j & 4) + d_p[d_c[i,5],10] * (j & 2) + d_p[d_c[i,6],10] * (j & 1))         #cg
                    and (0 == d_p[d_c[i,0],11] * (j & 64) + d_p[d_c[i,1],11] * (j & 32) + d_p[d_c[i,2],11] * (j & 16) + d_p[d_c[i,3],11] * (j & 8) + d_p[d_c[i,4],11] * (j & 4) + d_p[d_c[i,5],11] * (j & 2) + d_p[d_c[i,6],11] * (j & 1))         #ch
                    and (0 == d_p[d_c[i,0],12] * (j & 64) + d_p[d_c[i,1],12] * (j & 32) + d_p[d_c[i,2],12] * (j & 16) + d_p[d_c[i,3],12] * (j & 8) + d_p[d_c[i,4],12] * (j & 4) + d_p[d_c[i,5],12] * (j & 2) + d_p[d_c[i,6],12] * (j & 1))         #de
                    and (0 == d_p[d_c[i,0],13] * (j & 64) + d_p[d_c[i,1],13] * (j & 32) + d_p[d_c[i,2],13] * (j & 16) + d_p[d_c[i,3],13] * (j & 8) + d_p[d_c[i,4],13] * (j & 4) + d_p[d_c[i,5],13] * (j & 2) + d_p[d_c[i,6],13] * (j & 1))         #df
                    and (1 == d_p[d_c[i,0],14] * (j & 64) + d_p[d_c[i,1],14] * (j & 32) + d_p[d_c[i,2],14] * (j & 16) + d_p[d_c[i,3],14] * (j & 8) + d_p[d_c[i,4],14] * (j & 4) + d_p[d_c[i,5],14] * (j & 2) + d_p[d_c[i,6],14] * (j & 1))         #dg
                    and (0 == d_p[d_c[i,0],15] * (j & 64) + d_p[d_c[i,1],15] * (j & 32) + d_p[d_c[i,2],15] * (j & 16) + d_p[d_c[i,3],15] * (j & 8) + d_p[d_c[i,4],15] * (j & 4) + d_p[d_c[i,5],15] * (j & 2) + d_p[d_c[i,6],15] * (j & 1))):       #dh
                stat = True

    if stat:
        stat = False        
        for j in range(128):
            if (        (0 == d_p[d_c[i,0],0] * (j & 64) + d_p[d_c[i,1],0] * (j & 32) + d_p[d_c[i,2],0] * (j & 16) + d_p[d_c[i,3],0] * (j & 8) + d_p[d_c[i,4],0] * (j & 4) + d_p[d_c[i,5],0] * (j & 2) + d_p[d_c[i,6],0] * (j & 1))            #ae
                    and (0 == d_p[d_c[i,0],1] * (j & 64) + d_p[d_c[i,1],1] * (j & 32) + d_p[d_c[i,2],1] * (j & 16) + d_p[d_c[i,3],1] * (j & 8) + d_p[d_c[i,4],1] * (j & 4) + d_p[d_c[i,5],1] * (j & 2) + d_p[d_c[i,6],1] * (j & 1))            #af
                    and (0 == d_p[d_c[i,0],2] * (j & 64) + d_p[d_c[i,1],2] * (j & 32) + d_p[d_c[i,2],2] * (j & 16) + d_p[d_c[i,3],2] * (j & 8) + d_p[d_c[i,4],2] * (j & 4) + d_p[d_c[i,5],2] * (j & 2) + d_p[d_c[i,6],2] * (j & 1))            #ag
                    and (0 == d_p[d_c[i,0],3] * (j & 64) + d_p[d_c[i,1],3] * (j & 32) + d_p[d_c[i,2],3] * (j & 16) + d_p[d_c[i,3],3] * (j & 8) + d_p[d_c[i,4],3] * (j & 4) + d_p[d_c[i,5],3] * (j & 2) + d_p[d_c[i,6],3] * (j & 1))            #ah
                    and (0 == d_p[d_c[i,0],4] * (j & 64) + d_p[d_c[i,1],4] * (j & 32) + d_p[d_c[i,2],4] * (j & 16) + d_p[d_c[i,3],4] * (j & 8) + d_p[d_c[i,4],4] * (j & 4) + d_p[d_c[i,5],4] * (j & 2) + d_p[d_c[i,6],4] * (j & 1))            #be
                    and (0 == d_p[d_c[i,0],5] * (j & 64) + d_p[d_c[i,1],5] * (j & 32) + d_p[d_c[i,2],5] * (j & 16) + d_p[d_c[i,3],5] * (j & 8) + d_p[d_c[i,4],5] * (j & 4) + d_p[d_c[i,5],5] * (j & 2) + d_p[d_c[i,6],5] * (j & 1))            #bf
                    and (0 == d_p[d_c[i,0],6] * (j & 64) + d_p[d_c[i,1],6] * (j & 32) + d_p[d_c[i,2],6] * (j & 16) + d_p[d_c[i,3],6] * (j & 8) + d_p[d_c[i,4],6] * (j & 4) + d_p[d_c[i,5],6] * (j & 2) + d_p[d_c[i,6],6] * (j & 1))            #bg
                    and (0 == d_p[d_c[i,0],7] * (j & 64) + d_p[d_c[i,1],7] * (j & 32) + d_p[d_c[i,2],7] * (j & 16) + d_p[d_c[i,3],7] * (j & 8) + d_p[d_c[i,4],7] * (j & 4) + d_p[d_c[i,5],7] * (j & 2) + d_p[d_c[i,6],7] * (j & 1))            #bh
                    and (0 == d_p[d_c[i,0],8] * (j & 64) + d_p[d_c[i,1],8] * (j & 32) + d_p[d_c[i,2],8] * (j & 16) + d_p[d_c[i,3],8] * (j & 8) + d_p[d_c[i,4],8] * (j & 4) + d_p[d_c[i,5],8] * (j & 2) + d_p[d_c[i,6],8] * (j & 1))            #ce
                    and (1 == d_p[d_c[i,0],9] * (j & 64) + d_p[d_c[i,1],9] * (j & 32) + d_p[d_c[i,2],9] * (j & 16) + d_p[d_c[i,3],9] * (j & 8) + d_p[d_c[i,4],9] * (j & 4) + d_p[d_c[i,5],9] * (j & 2) + d_p[d_c[i,6],9] * (j & 1))            #cf
                    and (0 == d_p[d_c[i,0],10] * (j & 64) + d_p[d_c[i,1],10] * (j & 32) + d_p[d_c[i,2],10] * (j & 16) + d_p[d_c[i,3],10] * (j & 8) + d_p[d_c[i,4],10] * (j & 4) + d_p[d_c[i,5],10] * (j & 2) + d_p[d_c[i,6],10] * (j & 1))         #cg
                    and (0 == d_p[d_c[i,0],11] * (j & 64) + d_p[d_c[i,1],11] * (j & 32) + d_p[d_c[i,2],11] * (j & 16) + d_p[d_c[i,3],11] * (j & 8) + d_p[d_c[i,4],11] * (j & 4) + d_p[d_c[i,5],11] * (j & 2) + d_p[d_c[i,6],11] * (j & 1))         #ch
                    and (0 == d_p[d_c[i,0],12] * (j & 64) + d_p[d_c[i,1],12] * (j & 32) + d_p[d_c[i,2],12] * (j & 16) + d_p[d_c[i,3],12] * (j & 8) + d_p[d_c[i,4],12] * (j & 4) + d_p[d_c[i,5],12] * (j & 2) + d_p[d_c[i,6],12] * (j & 1))         #de
                    and (0 == d_p[d_c[i,0],13] * (j & 64) + d_p[d_c[i,1],13] * (j & 32) + d_p[d_c[i,2],13] * (j & 16) + d_p[d_c[i,3],13] * (j & 8) + d_p[d_c[i,4],13] * (j & 4) + d_p[d_c[i,5],13] * (j & 2) + d_p[d_c[i,6],13] * (j & 1))         #df
                    and (0 == d_p[d_c[i,0],14] * (j & 64) + d_p[d_c[i,1],14] * (j & 32) + d_p[d_c[i,2],14] * (j & 16) + d_p[d_c[i,3],14] * (j & 8) + d_p[d_c[i,4],14] * (j & 4) + d_p[d_c[i,5],14] * (j & 2) + d_p[d_c[i,6],14] * (j & 1))         #dg
                    and (1 == d_p[d_c[i,0],15] * (j & 64) + d_p[d_c[i,1],15] * (j & 32) + d_p[d_c[i,2],15] * (j & 16) + d_p[d_c[i,3],15] * (j & 8) + d_p[d_c[i,4],15] * (j & 4) + d_p[d_c[i,5],15] * (j & 2) + d_p[d_c[i,6],15] * (j & 1))):       #dh
                stat = True
    
    d_r[i] = stat


def run():
    d_p = cuda.to_device(p)
    
    c = np.array([getcandidate(0)])
    d_c = cuda.to_device(c)

    cnt = BLOKS * THREADS
    r = np.array([False]*cnt)
    d_r = cuda.to_device(r)

    calculate[BLOKS, THREADS](d_p, d_c, d_r)

    d_r.copy_to_host(r)
    print r[0]