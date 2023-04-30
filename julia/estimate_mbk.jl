using LinearAlgebra
using OrdinaryDiffEq
using Plots
using LaTeXStrings
using MAT
using ForwardDiff

file = matopen("../hardware/estimation_station3_new.mat")
data = read(file, "data")
close(file)

m, b, k = (data[4,end], data[5, end], 0.0)
tend = data[1,end]*1/2
A, φ = (0.3, 180*π/180)
n_freq = 3
ω(t) = π*sum( (1-(k-1)/n_freq)*sin(k*t) for k in 1:n_freq )
# ω(t) = 2*π*0.3*t
c, λ = (1, 4)
Γ = diagm(0=>[1.0, 1.0, 0.1])

# Sine of sines
xref(t) = A*sin(ω(t) + φ)

# # Parabola
# xref(t) = 1/2*t*t

# # Triangle
# period = 1.0
# xref(t) = 2*abs(t/period - floor(t/period+1/2))

xrefdot(t) = ForwardDiff.derivative(xref, t)
xrefddot(t) = ForwardDiff.derivative(xrefdot, t)

xtilde(x, t) = x[1] - xref(t)
xtildedot(x, t) = x[2] - xrefdot(t)

r(x, t) = xtildedot(x, t) + λ*xtilde(x, t)
v(x, t) = xrefdot(t) - λ*xtilde(x, t)
a(x, t) = xrefddot(t) - λ*xtildedot(x, t)
Y(x, t) = [a(x, t), v(x, t), x[1]]
u(x, t) = dot(Y(x,t), x[3:5]) - c*r(x, t)

function eom!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = 1/m*(u(x, t) - b*x[2] - k*x[1])
#     dx[3] = (x[3] > 0 ? f(x,t) : 1)
#     dx[4] = (x[4] > 0 ? g(x,t) : 1)
#     dx[5] = (x[5] > 0 ? h(x,t) : 1)
    dx[3:5] = -Γ\Y(x, t)*r(x,t)
end

x0 = [0, 0, -0.5, -0.25, -1.0]
tspan = (0, tend)
prob = ODEProblem(eom!, x0, tspan, saveat=range(tspan[1]; stop=tspan[2], length=10001))
sol = solve(prob, Tsit5())

x = [getindex.(sol.u, 1), getindex.(sol.u, 2), getindex.(sol.u, 3), getindex.(sol.u, 4), getindex.(sol.u, 5)]

p = plot(sol.t, xtilde.(sol.u, sol.t), linewidth=2, label=L"\tilde{x}", legendfontsize=15)
plot!(sol.t, xtildedot.(sol.u, sol.t), linewidth=2, label=L"\frac{d\tilde{x}}{dt}", legendfontsize=15)
# p = plot(sol.t, getindex.(sol.u, 1), linewidth=2, label=L"x", legendfontsize=15)
# plot!(sol.t, getindex.(sol.u, 1), linewidth=2, label=L"\frac{dx}{dt}", legendfontsize=15)
plot!(sol.t, getindex.(sol.u, 3), linewidth=2, label=L"\hat{m}", legendfontsize=15)
plot!(sol.t, getindex.(sol.u, 4), linewidth=2, label=L"\hat{b}", legendfontsize=15)
plot!(sol.t, getindex.(sol.u, 5), linewidth=2, label=L"\hat{k}", legendfontsize=15, legend=:bottomright)
# savefig(p, "adaptationrule1.pdf")
# savefig(p, "adaptationrule1.png")

p = plot(sol.t, xtilde.(sol.u, sol.t), linewidth=2, label=L"\tilde{x}", legendfontsize=12)
plot!(sol.t, xtildedot.(sol.u, sol.t), linewidth=2, label=L"\frac{d\tilde{x}}{dt}", legendfontsize=12)
plot!(sol.t, m.-getindex.(sol.u, 3), linewidth=2, label=L"\tilde{m}", legendfontsize=12)
plot!(sol.t, b.-getindex.(sol.u, 4), linewidth=2, label=L"\tilde{b}", legendfontsize=12)
plot!(sol.t, k.-getindex.(sol.u, 5), linewidth=2, label=L"\tilde{k}", legendfontsize=12)
# savefig(p, "adaptationrule2.pdf")
# savefig(p, "adaptationrule2.png")

# p = plot(sol.t, xtilde.(sol.u, sol.t), linewidth=1, label=L"\tilde{x}", legendfontsize=12)
# plot!(sol.t, xtildedot.(sol.u, sol.t), linewidth=1, label=L"\frac{d\tilde{x}}{dt}", legendfontsize=12)

# p = plot(xtilde.(sol.u, sol.t), xtildedot.(sol.u, sol.t), linewidth=1)

# p = plot(sol.t, xtilde.(sol.u, sol.t), linewidth=2, label=L"\tilde{x}: sim", legendfontsize=15)
# plot!(data[1,:], data[2,:], linewidth=2, label=L"\tilde{x}: real", linestyle=:dash, legendfontsize=15)
# savefig(p, "./TeX/figures/xtilde_real_sim.pdf")
# 
# p = plot(sol.t, xtildedot.(sol.u, sol.t), linewidth=2, label=L"\frac{d\tilde{x}}{dt}: sim", legendfontsize=15)
# plot!(data[1,:], data[3,:], linewidth=2, label=L"\frac{d\tilde{x}}{dt}: real", linestyle=:dash, legendfontsize=15)
# savefig(p, "./TeX/figures/xtildedot_real_sim.pdf")
# 
# p = plot(sol.t, getindex.(sol.u, 4), linewidth=2, label=L"\hat{m}: sim", legendfontsize=15)
# plot!(data[1,:], data[4,:], linewidth=2, label=L"\hat{m}: real", linestyle=:dash, legendfontsize=15)
# savefig(p, "./TeX/figures/mhat_real_sim.pdf")
# 
# p = plot(sol.t, getindex.(sol.u, 3), linewidth=2, label=L"\hat{b}: sim", legendfontsize=15)
# plot!(data[1,:], data[5,:], linewidth=2, label=L"\hat{b}: real", linestyle=:dash, legendfontsize=15)
# savefig(p, "./TeX/figures/bhat_real_sim.pdf")
