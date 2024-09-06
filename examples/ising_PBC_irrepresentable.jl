using MPSKit, MPSKitModels, TensorKit
using MKL, LinearAlgebra

BLAS.set_num_threads(2)

MPSKit.Defaults.set_parallelization("sites" => false, "transfers" => true, "derivatives" => true)

using Plots # for demonstration purposes

L = 1000 # length of the chain
D = 8 # bonddimension

physical = Rep[ℤ₂](0 => 1, 1 => 1)
virtual  = Rep[ℤ₂](0 => D//2, 1 => D//2)
init_state = FiniteMPS(L, physical, virtual);

H = periodic_boundary_conditions(transverse_field_ising(  Z2Irrep; g=1.0), L);

groundstate, environment, δ = find_groundstate(
  init_state, H, DMRG(
    maxiter = 1000,
    tol = 1e-10/L,);
  
);

