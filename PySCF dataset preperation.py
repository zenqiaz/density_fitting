import numpy as np
from numpy import array
import pyscf
#from gpu4pyscf.dft import rks
from pyscf import gto, scf, mcscf
from collections import namedtuple
from pyscf.mcscf import addons
import multiprocessing
import copy
import pickle
np.set_printoptions(threshold=np.inf)

from pathos.multiprocessing import ProcessingPool as Pool
#from pprint import pprint

#Geometry optimization cycle 7
#Cartesian coordinates (Angstrom)
# Atom        New coordinates             dX        dY        dZ
#   C  -0.000000   0.000004  -0.155732   -0.000000  0.000133  0.000205
#   O   0.000000  -0.000003   1.028619    0.000000 -0.000601  0.000246
#   H   0.000000   0.924238  -0.736448    0.000000 -0.000105 -0.000866
#   H   0.000000  -0.924239  -0.736440    0.000000  0.000573  0.000415
#ANG here


#>>> for atom in opt_mol.atom_coords():
#...     print(atom)
#...
#[-1.38020777e-12  8.40590542e-06 -2.94290677e-01]
#[ 5.54806035e-13 -6.45296814e-06  1.94380867e+00]
#[ 4.33183232e-13  1.74655626e+00 -1.39168459e+00]
#[ 3.83288134e-13 -1.74655821e+00 -1.39166908e+00]
#Bohr here


configuration = [['C', (0.0, 0.0, -0.155732)],['O', np.array([0.0, 0.0, 1.028619])],['H', np.array([0.0, 0.924238, -0.736448])],['H', np.array([0.0, -0.924239, -0.736448])]]
scatter_mesh = []
H_S = [[0,0,0],[0,0.1,0],[0,-0.1,0],[0,0.05,0.073],[0,-0.05,0.073],[0,-0.05,-0.073],[0,0.05,-0.073]]
O_S = [[0,0,0],[0,0.1,0],[0,-0.1,0],[0,0,0.1],[0,0,-0.1],[0,0.07,0.07],[0,0.07,-0.07],[0,-0.07,0.07],[0,-0.07,-0.07],[0.02,0,0],[0.02,0.1,0],[0.02,-0.1,0],[0.02,0,0.1],[0.02,0,-0.1],[0.02,0.07,0.07],[0.02,0.07,-0.07],[0.02,-0.07,0.07],[0.02,-0.07,-0.07]]
H_array = []
scatter_mesh = []
O_array = []
for i in H_S:
    H_array.append(np.array(i))

for i in O_S:
    O_array.append(np.array(i))

scatter_mesh.append(np.array(H_array))
scatter_mesh.append(np.array(O_array))
#scatter_mesh[0] /= 2
#scatter_mesh[1] /= 2

def molecular_directory(name: str, configuration: list):#with symmetry == True, rdm2 is not available
    Input = namedtuple('Input',('name', 'atom', 'basis', 'multiplicity', 'charge', 'number_electrons'))
    my_input = Input('HCHO', configuration, 'sto-3g', 1, 0, 16)
    
    mol = gto.M(
        atom=my_input.atom,
        basis=my_input.basis,
        unit="ANG",
        symmetry=False,
    )
    mol.build()
    return mol

def create_example_scatter(name, configuration, scatter):
    for i in range(3):
        configuration[i+1]+= scatter[i] 
    return molecular_directory(configuration)


def create_input_set(initial_config: list, scatter_mesh: list, name: str):
    input_set = []
    count = 0
    for i in scatter_mesh[0]:
        for j in scatter_mesh[1]:
            for k in scatter_mesh[1]:
                cur_config = copy.deepcopy(initial_config)
                cur_config[1][1] += i
                cur_config[1][1] = tuple(cur_config[1][1].astype(float))
                cur_config[2][1] += j
                cur_config[2][1] = tuple(cur_config[2][1].astype(float))
                cur_config[3][1] += k
                cur_config[3][1] = tuple(cur_config[3][1].astype(float))
                input_set.append(molecular_directory(name+'_'+str(count), cur_config))
                count += 1
    return input_set

def prepare_sample(molecule):
    ncas = 8
    nelecas=8
    mf = scf.RHF(molecule).run()
    mcas = mcscf.CASSCF(mf,ncas,nelecas).run()
    mo_coeff = mcas.mo_coeff
    #mo_occ = mcas.mo_occ
    rdm1,rdm2 = mcscf.addons.make_rdm12(mcas)
    return (mo_coeff, rdm1, rdm2)

def prepare_sample_raw(molecule):
    ncas = 8
    nelecas=8
    mf = scf.RHF(molecule).run()
    return mf

def prepare_sample_reshape(molecule):
    ncas = 8
    nelecas=8
    mf = scf.RHF(molecule).run()
    mcas = mcscf.CASSCF(mf,ncas,nelecas).run()
    mo_coeff = mcas.mo_coeff
    length = len(mo_coeff)
    #mo_occ = mcas.mo_occ
    rdm1,rdm2 = mcscf.addons.make_rdm12(mcas)
    return (mcas.converged, mo_coeff, rdm1, rdm2.reshape(length**4))

def prepare_sample_tolist(molecule):
    ncas = 8
    nelecas=8
    mf = scf.RHF(molecule).run()
    mcas = mcscf.CASSCF(mf,ncas,nelecas).run()
    mo_coeff = mcas.mo_coeff
    #mo_occ = mcas.mo_occ
    rdm1,rdm2 = mcscf.addons.make_rdm12(mcas)
    return (mo_coeff.tolist(), rdm1.tolist(), rdm2.tolist())

def run_with_pool(input_set):
    #start_time = time.time()
    pool = Pool(10)  # Use 10 parallel workers
    tasks = input_set  # 100 tasks
    results = pool.map(prepare_sample, tasks, chunksize = 10)  # Run tasks in parallel
    pool.close()  # Close the pool
    pool.join()   # Wait for all workers to finish
    #end_time = time.time()
    #print(f"Pool execution time: {end_time - start_time:.2f} seconds")
    return results

def run_with_loop(input_set):
    #start_time = time.time()
    
    tasks = input_set  # 100 tasks
    results = [prepare_sample_reshape(task_id) for task_id in tasks]  # Run tasks sequentially
    
    #end_time = time.time()
    #print(f"Loop execution time: {end_time - start_time:.2f} seconds")
    return results

test_mesh = [[np.array([0,0,0]),np.array([0,0,0.1])],[np.array([0,0,0]),np.array([0,0,0.1]),np.array([0,0.1,0])]]
input_set = create_input_set(configuration, scatter_mesh = test_mesh, name = 'HCHO')#generate a test set
#input_set = create_input_set(configuration, scatter_mesh = scatter_mesh, name = 'HCHO')#use this line to generate the full set
print(input_set[0].atom_coords(unit='Angstrom'))
#print("Number of electrons:", input_set[0].nelectron)
ncas=8#1.47s
nelecas=8
mf = scf.RHF(input_set[-2]).run()
mcas = mcscf.CASSCF(mf,ncas,nelecas).run()
mo_coeff = mcas.mo_coeff
print(mo_coeff)
mo_occ = mcas.mo_occ
print(mo_occ)
#rdm2 = mcas.make_rdm2()
#rdm2 = make_rdm2(mo_coeff, mo_occ)
#rdm2 = scf.hf.make_rdm2(mo_coeff, mo_occ)
#rdm2 = mcscf.addons.make_rdm12(mcas)
rdm1,rdm2 = mcscf.addons.make_rdm12(mcas)
#rdm2 = addons.make_rdm2(mcas)
print(prepare_sample(input_set[0]))
#dataset = run_with_pool(input_set)
dataset = run_with_loop(input_set)
#with open('output.txt', 'w') as file:
 #   for item in dataset:
  #      file.write(f"{item}\n")


def run_with_loop_write_origin(input_set, file_name):
    #start_time = time.time()
    
    tasks = input_set  # 100 tasks
    with open(file_name, 'w') as file:
        for task_id in tasks:
            file.write(f"{prepare_sample(task_id)}\n")
    #results = [prepare_sample(task_id) for task_id in tasks]  # Run tasks sequentially
    
    #end_time = time.time()
    #print(f"Loop execution time: {end_time - start_time:.2f} seconds")
    return 


def run_with_loop_write_old(input_set, file_name):
    #start_time = time.time()
     
    tasks = input_set  # 100 tasks
    with open(file_name, 'w') as file:
        file.write(f"from numpy import array\n")
        file.write(f"dataset = [\n")
        for task_id in range(len(tasks)):
            if task_id > 0:
                file.write(',')
            res = prepare_sample_reshape(tasks[task_id])
            if res[0]:
                file.write(f"{res[1:]}")
            #break
        file.write(f"]\n")
    #results = [prepare_sample(task_id) for task_id in tasks]  # Run tasks sequentially
    
    #end_time = time.time()
    #print(f"Loop execution time: {end_time - start_time:.2f} seconds")
    return 

def run_with_loop_write(input_set, file_name):
    #start_time = time.time()
     
    tasks = input_set  # 100 tasks
    with open(file_name, 'w') as file:
        file.write(f"from numpy import array\n")
        file.write(f"dataset = [\n")
        for task_id in range(len(tasks)):
            mf = prepare_sample_raw(tasks[task_id])
            mcas = mcscf.CASSCF(mf,ncas,nelecas).run()
            mo_coeff = mcas.mo_coeff
            rdm1,rdm2 = mcscf.addons.make_rdm12(mcas)
            length = len(mo_coeff)
            if mf.converged:
                if task_id > 0:
                    file.write(',')
                file.write(f"{(mo_coeff, rdm1, rdm2.reshape(length**4))}")
            #break
        file.write(f"]\n")
    #results = [prepare_sample(task_id) for task_id in tasks]  # Run tasks sequentially
    
    #end_time = time.time()
    #print(f"Loop execution time: {end_time - start_time:.2f} seconds")
    return 



def run_with_loop_pkl(input_set, file_name):
    #start_time = time.time()
     
    tasks = input_set  # 100 tasks
    with open(file_name, 'wb') as f:
        for task_id in range(len(tasks)):
            mf = prepare_sample_raw(tasks[task_id])
            mcas = mcscf.CASSCF(mf,ncas,nelecas).run()
            mo_coeff = mcas.mo_coeff
            rdm1,rdm2 = mcscf.addons.make_rdm12(mcas)
            if mf.converged:
                pickle.dump((mo_coeff, rdm1, rdm2), f)
    #results = [prepare_sample(task_id) for task_id in tasks]  # Run tasks sequentially
    
    #end_time = time.time()
    #print(f"Loop execution time: {end_time - start_time:.2f} seconds")
    return 

#run_with_loop_write(input_set, 'output2.py')
#from output2 import dataset

run_with_loop_pkl(input_set, 'RDM2_dataset.pkl')
#with open('test.pkl', 'rb') as f:
 #   loaded_data = pickle.load(f)


loaded_data = []
with open('RDM2_dataset.pkl', 'rb') as f:
    while True:
        try:
            t = pickle.load(f)  # Load one tuple at a time
            loaded_data.append(t)
        except EOFError:
            break  # End of file reached

print(f"Loaded {len(loaded_data)} tuples.")
print(f"Shape of first tuple's third element: {loaded_data[0][2].shape}")

# Specify the file path
file_path = 'test.pkl'

# Get the file size in bytes
file_size = os.path.getsize(file_path)

# Convert to KB or MB
file_size_kb = file_size / 1024
file_size_mb = file_size / (1024 * 1024)
print(f"File size: {file_size} bytes ({file_size_kb:.2f} KB, {file_size_mb:.2f} MB)")