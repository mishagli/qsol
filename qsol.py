"""
Dmytro Mishagli, UCD

04 Dec 2019 -- the script was created.


"""

import numpy as np

def basis(x,n,L):
	'''
	The basis function.
	'''
	return np.sqrt(2/L) * np.sin( x * n * np.pi / L )

def integ(n,m,lower_limit,upper_limit,L):
	"""
	Returns values of the integrals in a Hamiltonian of a square potential well,
	calculated analytically
	"""
	alpha = n*np.pi/L
	beta  = m*np.pi/L

	if n!=m:
		f = lambda x: 0.5 * ( np.sin((alpha-beta)*x)/(alpha-beta) - np.sin((alpha+beta)*x)/(alpha+beta) )
		return 2/L * (f(upper_limit) - f(lower_limit))
	else:
		f = lambda x: 0.5 * ( x - np.sin((alpha+beta)*x)/(alpha+beta) )
		return 2/L * (f(upper_limit) - f(lower_limit))

def potential_piecewise(n,m,Vext,Vint,wells,barriers):
	'''
	Vext - list, left and right (exterior) potential wells' heights
	Vint - list, interatomic wells heights
	wells - list, iterior wells' widths
	barriers - list, distances between the wells
	'''
	# number of wells
	nWells = len(wells)
	# add zero element to the begin of a list of barriers
	barriers = [0] + barriers
	# size of shift from 0
	h = np.sum(wells) + np.sum(barriers)
	# width of an infinite square well
	L = 3*h
	# initialise variables
	s = 0
	lower_limit = h
	upper_limit = h
	# iterate through the square well (sequence of wells)
	for i in range(1,nWells):
		lower_limit += wells[i] + barriers[i-1]
		upper_limit += wells[i] + barriers[i]
		s += Vint[i-1] * integ( n,m, lower_limit, upper_limit, L )
	return Vext[0]*integ(n, m, 0, h, L) + Vext[1]*integ(n, m, L-h, L, L) + s

def get_solution(Vext,Vint,wells,barriers,num_bas_funs):
	# size of shift from 0
	h = np.sum(wells) + np.sum(barriers)
	# width of an infinite square well
	L = 3*h
	# compute a Hamiltonian (n,m) square matrix
	potential_matrix = [[potential_piecewise(n,m,Vext,Vint,wells,barriers) for n in range(1,num_bas_funs+1)] for m in range(1,num_bas_funs+1)]
	# compute a list of eigenvalues for H0
	evalues = [n**2*np.pi**2/(L**2) for n in range(1,num_bas_funs+1)]
	# create a diagonal matrix
	H0 = np.diag(evalues)
	# get solution
	eigvals, eigvecs = np.linalg.eigh(H0+potential_matrix)

	# bound states are below the exterior height
	eigvals = eigvals[eigvals<min(Vext)]
	# transopse the matrix with eigenvectors (we need rows)
	# and left only those who correspond to the bound states
	eigvecs = eigvecs.T[:len(eigvals)]

	return eigvals, eigvecs


