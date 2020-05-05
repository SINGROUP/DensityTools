import numpy as np
from ase.io import read, write
import sys

def main(filename):
    atoms = read(filename, format='vasp')
    with open(filename, 'r') as f:
        n_line = 0
        while True:
            line = f.readline()
            n_line += 1
            if line.strip() == '':
                break
        N_voxl = np.array(list(map(int, f.readline().split())))
        n_line += 1
        tot_voxl = np.prod(N_voxl)
        data = np.zeros((tot_voxl))
        count = 0
        while count < tot_voxl:
            line = np.array(list(map(float, f.readline().split())))
            n = line.shape[0]
            data[count:count+n] = line
            count += n
    data = data.reshape(N_voxl, order='F')
    # scaling vasp 
    data /= atoms.cell.volume
    write(f'{filename}.cube', atoms, data=data)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise TypeError(f'Usage: {sys.argv[0]} <filename1> ...')
    
    for filename in sys.argv[1:]:
        main(filename)

