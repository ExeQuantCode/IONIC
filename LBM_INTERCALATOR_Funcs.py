# import jax.numpy as np
import sys
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import subprocess
from scipy.sparse import csr_matrix, csc_matrix #Sparse matrix library
from scipy.sparse.linalg import bicgstab, spilu, LinearOperator # Sparse linear algebra library
from PIL import Image

def charge_density(grid_size, electrons, concentration, constants):
    """
    Calculate charge density from electron and ion concentration.
    Parameters
    ----------
    grid_size : tuple of int
        Size of the grid (NY, NX).
    electrons : ndarray of float, shape (NY, NX)
        Electron concentration in the grid.
    concentration : ndarray of float, shape (NY, NX)
        Ion concentration in the grid.
    Returns
    -------
    charge_density : ndarray of float, shape (NY, NX)
        Charge density in the grid.
    """
    NX, NY = grid_size[1], grid_size[0]
    F = constants['F']  # Faraday constant
    dx_real = constants['dx_real']  # Real space grid spacing in meters
    # Initialize charge density
    charge_density = np.zeros((NY, NX))
    charge_density += -F * electrons * (dx_real**3)
    charge_density += F * concentration * (dx_real**3)
    return charge_density

def find_interface(solid_mask):
	"""
	Find interface nodes: solid nodes that have at least one electrolyte neighbor.
	
	Parameters
	----------
	solid_mask : ndarray of bool, shape (NY, NX)
		Mask where True indicates solid phase
		
	Returns
	-------
	interface_mask : ndarray of bool, shape (NY, NX)
		Mask where True indicates interface nodes
	"""
	interface_mask = np.zeros_like(solid_mask, dtype=bool)
	
	for i in range(1, solid_mask.shape[0] - 1):
		for j in range(1, solid_mask.shape[1] - 1):
			if solid_mask[i, j]:
				# Check 4-connected neighbors (could extend to 8-connected)
				neighbors = [
					solid_mask[i-1, j],  # up
					solid_mask[i+1, j],  # down  
					solid_mask[i, j-1],  # left
					solid_mask[i, j+1]   # right
				]
				
				# Interface if any neighbor is not solid (i.e., electrolyte)
				if not all(neighbors):
					interface_mask[i, j] = True
				# end if
			# end if
		# end for
	# end for
	
	return interface_mask

def sparse_matrix_2d(NY, NX, dx=1.0, boundary_conditions=None):
	"""
	Create sparse matrix for 2D Poisson equation: ∇²φ = -ρ/ε
	
	Parameters:
	-----------
	NY, NX : int
		Grid dimensions (rows, columns)
	dx : float
		Grid spacing
	boundary_conditions : dict, optional
		Boundary condition specifications
		
	Returns:
	--------
	scipy.sparse matrix in CSC format
	"""
	N_total = NY * NX
	
	# Estimate number of non-zero entries (interior: 5, boundary: 3-4)
	nnz_estimate = 5 * N_total
	
	row_indices = []
	col_indices = []
	values = []
	
	def add_entry(row, col, val):
		row_indices.append(row)
		col_indices.append(col)
		values.append(val)
	
	# Build matrix row by row
	for idx in range(N_total):
		i = idx // NX  # row index
		j = idx % NX   # column index
		
		# Diagonal coefficient (will be modified by stencil)
		diagonal_coeff = 0.0
		
		# 5-point stencil coefficients
		coeff = 1.0 / (dx * dx)
		
		# Up neighbor (i-1, j)
		if i > 0:
			neighbor_idx = (i-1) * NX + j
			add_entry(idx, neighbor_idx, coeff)
			diagonal_coeff -= coeff
		else:
			# Top boundary - homogeneous Dirichlet: φ = 0
			diagonal_coeff -= coeff
			
		# Down neighbor (i+1, j)  
		if i < NY - 1:
			neighbor_idx = (i+1) * NX + j
			add_entry(idx, neighbor_idx, coeff)
			diagonal_coeff -= coeff
		else:
			# Bottom boundary - homogeneous Dirichlet: φ = 0
			diagonal_coeff -= coeff
			
		# Left neighbor (i, j-1)
		if j > 0:
			neighbor_idx = i * NX + (j-1)
			add_entry(idx, neighbor_idx, coeff)
			diagonal_coeff -= coeff
		else:
			# Left boundary - homogeneous Dirichlet: φ = 0
			diagonal_coeff -= coeff
			
		# Right neighbor (i, j+1)
		if j < NX - 1:
			neighbor_idx = i * NX + (j+1)
			add_entry(idx, neighbor_idx, coeff)
			diagonal_coeff -= coeff
		else:
			# Right boundary - homogeneous Dirichlet: φ = 0
			diagonal_coeff -= coeff
			
		# Add diagonal entry
		add_entry(idx, idx, diagonal_coeff)
	
	# Create sparse matrix in CSC format (good for iterative solvers)
	matrix = csc_matrix((values, (row_indices, col_indices)), 
						shape=(N_total, N_total), dtype=np.float64)
	
	return matrix

def apply_boundary_conditions_2d(matrix, rhs, NY, NX, bc_dict=None):
	"""
	Apply boundary conditions by modifying matrix and RHS
	
	Parameters:
	-----------
	matrix : sparse matrix
		System matrix
	rhs : ndarray
		Right-hand side vector
	NY, NX : int
		Grid dimensions
	bc_dict : dict
		Boundary conditions specification
		
	Returns:
	--------
	matrix, rhs : modified system
	"""
	if bc_dict is None:
		# Default: homogeneous Dirichlet on all boundaries
		return matrix, rhs
	
	# Convert to LIL format for efficient modification
	A = matrix.tolil()
	b = rhs.copy()
	
	# Apply Dirichlet boundary conditions
	for boundary, (bc_type, value) in bc_dict.items():
		if bc_type == 'dirichlet':
			if boundary == 'top':
				for j in range(NX):
					idx = 0 * NX + j  # i=0
					A[idx, :] = 0
					A[idx, idx] = 1
					b[idx] = value
					
			elif boundary == 'bottom':
				for j in range(NX):
					idx = (NY-1) * NX + j  # i=NY-1
					A[idx, :] = 0
					A[idx, idx] = 1
					b[idx] = value
					
			elif boundary == 'left':
				for i in range(NY):
					idx = i * NX + 0  # j=0
					A[idx, :] = 0
					A[idx, idx] = 1
					b[idx] = value
					
			elif boundary == 'right':
				for i in range(NY):
					idx = i * NX + (NX-1)  # j=NX-1
					A[idx, :] = 0
					A[idx, idx] = 1
					b[idx] = value
			else:
				print(f"Warning: Unsupported boundary condition '{bc_type}' for {boundary}")
				# You can add more boundary conditions here if needed
				sys.exit(1)
		else:
			print(f"Warning: Unsupported boundary condition '{bc_type}' for {boundary}")
			# You can add more boundary conditions here if needed
			sys.exit(1)
	# end for

	
	return A.tocsc(), b

def PreConditioner(Pmatrix, tol=1e-5):
	ilu = spilu(Pmatrix, fill_factor=10, drop_tol=tol)
	M = LinearOperator((Pmatrix.shape[0], Pmatrix.shape[1]), matvec=ilu.solve)
	return M

def create_preconditioner(matrix, drop_tol=1e-6, fill_factor=20):
    """Create ILU preconditioner"""
    try:
        ilu = spilu(matrix.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
        M = LinearOperator(matrix.shape, matvec=ilu.solve)
        return M
    except:
        print("Warning: ILU preconditioner failed, using no preconditioning")
        return None

def solve_poisson_2d(charge_density, NY, NX, constants=None, A=None, initial=None, dx=1.0, epsilon=1.0, 
                     boundary_conditions=None, tol=1e-12, max_iter=10000):
    """
    Solve 2D Poisson equation: ∇²φ = -ρ/ε
    
    Parameters:
    -----------
    charge_density : ndarray, shape (NY, NX)
        Charge density field
    NY, NX : int
        Grid dimensions
    dx : float
        Grid spacing
    epsilon : float
        Permittivity
    boundary_conditions : dict
        Boundary condition specifications
    tol : float
        Solver tolerance
    max_iter : int
        Maximum iterations
        
    Returns:
    --------
    potential : ndarray, shape (NY, NX)
        Electric potential field
    info : dict
        Solver information
    """
    
    # Create system matrix
    if A is None:
        A = sparse_matrix_2d(NY, NX, dx)
    
    if constants is None:
        print('ERROR: constants not provided')
        sys.exit(1)
    
    # Scale epsilon by dx to dimensionless it
    epsilon = epsilon * constants['dx_real']

    # Create RHS vector
    rhs = (-charge_density.flatten() / epsilon)
    
    # Apply boundary conditions
    A, rhs = apply_boundary_conditions_2d(A, rhs, NY, NX, boundary_conditions)
    
    # Create preconditioner
    M = create_preconditioner(A)
    
    # Solve system
    # print(f"Solving {NY}x{NX} Poisson equation...")
    solution, convergence_flag = bicgstab(A, rhs, M=M, x0=initial, rtol=tol, maxiter=max_iter)
    
    # Check convergence
    if convergence_flag == 0:
        # print("✓ Solver converged successfully")
        pass
    else:
        print(f"⚠ Solver warning: convergence flag = {convergence_flag}")
    
    # Reshape solution back to 2D
    potential = solution.reshape((NY, NX))
    
    info = {
        'convergence_flag': convergence_flag,
        'matrix_shape': A.shape,
        'nnz': A.nnz,
        'condition_estimate': 'not computed'
    }
    
    return potential, info

#gradient of the potential in the directions of the vecolities
def grad_potential(potential):
	'''
	Calculate the gradient of the potential in the directions of the velocities.
	Args:
		potential: 2D numpy array of potential values
		e: 2D numpy array of velocity vectors
	'''
	
	delPot_xx = np.gradient(potential, axis=1)  # central-difference ∂φ/∂x
	delPot_yy = np.gradient(potential, axis=0)  # central-difference ∂φ/∂y
	
	return delPot_xx, delPot_yy

@njit
def equilibrium_function(c_i, w, e, grid_size=None):
    '''
    Calculate the equilibrium distribution from the concentration and velocity weights
    '''
    if grid_size is None:
        print("Warning: grid_size is None, using c_i shape")
        grid_size = c_i.shape
    #end if

    NY, NX = grid_size
    ge = np.zeros((len(e), NY, NX), dtype=np.float64)
    for a in range(len(e)):
        ge[a] = w[a] * c_i
    return ge


def stream_and_bounce(g, c_i, velocity_vectors, solid_mask, electrolyte_mask, boundary_mask, constants):
    """
    Perform D2Q9 streaming with boundary conditions applied in sequence:
    1. Stream all distributions
    2. Apply boundary bounce-back
    3. Apply interface conditions
    
    Parameters
    ----------
    g : ndarray, shape (9, NY, NX)
        Post-collision distribution
    c_i : ndarray, shape (NY, NX)
        Current concentration field
    solid_mask, electrolyte_mask, boundary_mask : ndarray of bool, shape (NY, NX)
        Phase masks
    
    Returns
    -------
    g_new : ndarray, shape (9, NY, NX)
        Distribution after streaming and boundary conditions
    """
    # Constants
    C_max = constants['C_max']
    e = velocity_vectors['e']
    opp_list = velocity_vectors['opp_list']
    
    # Step 1: Apply interface conditions
    interface_mask = find_interface(solid_mask)

    g = interface_condition(g, c_i, velocity_vectors, interface_mask, electrolyte_mask, C_max=C_max)

    # Step 2: Stream all distributions

    g_streamed = np.zeros_like(g)
    for a in range(len(e)):
        cx, cy = e[a]
        g_streamed[a] = np.roll(np.roll(g[a], cx, axis=1), cy, axis=0)
    
    # Step 3: Apply standard boundary conditions
    g_new = g_streamed.copy()
    
    # Bounce-back at domain boundaries
    for a in range(len(e)):
        g_new[opp_list[a]][boundary_mask] = g_streamed[a][boundary_mask]
    

    
    return g_new


def interface_condition(g, c_i, velocity_vectors, interface_mask, electrolyte_mask, C_max=100):
    """
    Apply interface condition to the distribution function g at the interface.
    """
    # Get the opposite directions for bounce-back
    opp_list = velocity_vectors['opp_list']
    # Get the velocity vectors
    e = velocity_vectors['e']
    # One way bounce-back condition at the interface
    for i in range(g.shape[1]):
        for j in range(g.shape[2]):
            if interface_mask[i, j]:  # Fixed variable name
                for a in range(len(e)):
                    cx, cy = e[a]
                    # Calculate neighbor indices
                    ni, nj = i + cy, j + cx  # Neighbor indices
                    
                    # Check bounds
                    if 0 <= ni < g.shape[1] and 0 <= nj < g.shape[2]:
                        if electrolyte_mask[ni, nj]:
                            # Interface is saturated - bounce back everything
                            if c_i[i, j] >= C_max:
                                # Complete bounce-back: swap distributions
                                temp = g[a, ni, nj]
                                g[a, ni, nj] += g[opp_list[a], ni, nj]
                                g[opp_list[a], ni, nj] = 0
                                
                            # Interface not saturated - allow partial diffusion
                            else:
                                # Calculate how much can be absorbed
                                available_capacity = C_max - c_i[i, j]
                                incoming_mass = g[opp_list[a], ni, nj]  # From electrolyte to interface
                                
                                if incoming_mass <= available_capacity:
                                    # Accept all incoming mass
                                    pass  # Normal streaming will handle this
                                else:
                                    # Accept only what fits, bounce back the rest
                                    accept_fraction = available_capacity / incoming_mass
                                    g[opp_list[a], ni, nj] *= accept_fraction
                                    g[a, ni, nj] += incoming_mass * (1 - accept_fraction)
    
    return g

# @njit
def collision_stream(g, c_i, w, velocity_vectors, J_i, tau, potential, solid_mask, electrolyte_mask, boundary_mask, constants):
    """
    One LBM step: collision → stream+on-link bounce → macroscopic update.

    Steps
    -----
    1) Collision: relax g toward local equilibrium ge.  
        g += -(g - ge) / tau_g

    2) Streaming + on-link bounce-back:  
        g = stream_and_bounce(g, electrolyte_mask)

    3) Macroscopic update:  
        c_i = sum_a g[a]

    Parameters
    ----------
    g : ndarray, shape (9, NY, NX)
        Current discrete distributions.
    c_i : ndarray, shape (NY, NX)
        Current concentration field.
    w : ndarray, shape (9,)
        D2Q9 weights.

    Returns
    -------
    g : ndarray, shape (9, NY, NX)
        Updated distributions.
    c_i : ndarray, shape (NY, NX)
        Updated concentration.
    """
    tau_g_s = tau['Solid']
    tau_g_e = tau['Electrolyte']
    z, F, R, T, dx_real = constants['z'], constants['F'], constants['R'], constants['T'], constants['dx_real']
    e = velocity_vectors['e']
    opp_list = velocity_vectors['opp_list']
    NY, NX = g.shape[1], g.shape[2]
    # 1) collision
    ge = equilibrium_function(c_i, J_i, e, grid_size=(NY, NX))
    grad_xx, grad_yy = grad_potential(potential)

    # Relaxation step towards equilibrium
    for a in range(len(e)):
        cx, cy = e[a]
        grad_proj = (cx * grad_xx + cy * grad_yy) * 1e-1
        
        # Solid Relaxation step towards equilibrium diffusion
        g[a][(solid_mask)] += -(g[a][(solid_mask)] - ge[a][(solid_mask)]) / tau_g_s

        # Electrolyte Relaxation step towards equilibrium diffusion 
        g[a][(electrolyte_mask)] += -(g[a][(electrolyte_mask)] - ge[a][(electrolyte_mask)]) / tau_g_e
        
        # Electrolyte Relaxation step towards equilibrium driven by electric potential
        g[a][electrolyte_mask] += ((z * F*dx_real) / (R * T)) * (grad_proj[electrolyte_mask]) * ge[a][electrolyte_mask]

    # 2) stream + bounce
    g = stream_and_bounce(g, c_i, velocity_vectors, solid_mask, electrolyte_mask, boundary_mask, constants)

    # 3) macroscopic
    c_i = np.sum(g, axis=0)
    # c_i = np.round(c_i, 12)  # round to 2 decimal places for clarity
    return g, c_i
