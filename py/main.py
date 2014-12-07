from numbapro import cuda
from polynom import p
import numpy as np



def main():
    d_p = cuda.to_device(p)
    

    print str(r[0:5])
    print str(r[-5:])
    
main()