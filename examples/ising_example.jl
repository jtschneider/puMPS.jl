using puMPS
using LinearAlgebra
using TensorKit, MPSKit

D = 8
N = 100

# M = rand_puMPState(ComplexF64, 2, D, N)

M2 = InfiniteMPS([Rep[ℤ₂](0 => 1, 1 => 1)], [Rep[ℤ₂](0 => D//2, 1 => D//2)])

M = M2.AC

H = ising_local_MPO(ComplexF64)

vumps_opt!(M, H, 1e-6, maxitr=5) #Pre-optimization using DMRG-like method
minimize_energy_local!(M, H, 1000, step=0.05)




println("Computing excitations!")

ks_tocompute = collect(-5:5)
num_states = [6,5,4,7,4,5,6]

ens, ks, exs = excitations!(M, ising_PBC_MPO_split(ComplexF64), ks_tocompute, num_states)

H1 = Hn_in_basis(M, ising_Hn_MPO_split(ComplexF64, 1, N), exs, ks_tocompute)
H2 = Hn_in_basis(M, ising_Hn_MPO_split(ComplexF64, 2, N), exs, ks_tocompute)

ind1 = argmin(real.(ens))
indT = argmax(abs.(H2[:,ind1]))

en1 = real(ens[ind1])
enT = real(ens[indT])

fac = real(2.0 / (enT-en1))

c = 2*abs2(H2[indT,ind1] * fac)
@show real(c)

using Plots

p = Plots.scatter(ks, real(ens), label="momentum")
