using LinearAlgebra
using OrdinaryDiffEq
# using Plots
using PyPlot
using LaTeXStrings
using MAT
using ForwardDiff

file = matopen("../hardware/estimation_station3_new.mat")
data = read(file, "data")
close(file)

m, b, k = (data[4,end], data[5, end], 0.0)
# m, b, k = (0.665, 1.2, 0.0)
θ = [m, b, k]
tend = data[1,end]*1/4
A, φ = (0.3, 180*π/180)
n_freq = 3
ω(t) = π*sum( (1-(k-1)/n_freq)*sin(k*t) for k in 1:n_freq )
# ω(t) = 2*π*0.3*t
c, λ = (0.1, 4.0)
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
V(x, t) = 1/2*m*r(x,t)^2 + c*λ*xtilde(x, t)^2 + 1/2*dot(x[3:5]-θ, Γ, x[3:5]-θ)
Vdot(x, t) = -c*λ*λ*xtilde(x, t)^2 - c*xtildedot(x,t)^2 - b*r(x,t)^2

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
sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10)

x = [getindex.(sol.u, 1), getindex.(sol.u, 2), getindex.(sol.u, 3), getindex.(sol.u, 4), getindex.(sol.u, 5)]

# p = plot(sol.t, xtilde.(sol.u, sol.t), linewidth=2, label=L"\tilde{x}", legendfontsize=15)
# plot!(sol.t, xtildedot.(sol.u, sol.t), linewidth=2, label=L"\frac{d\tilde{x}}{dt}", legendfontsize=15)
# # p = plot(sol.t, getindex.(sol.u, 1), linewidth=2, label=L"x", legendfontsize=15)
# # plot!(sol.t, getindex.(sol.u, 1), linewidth=2, label=L"\frac{dx}{dt}", legendfontsize=15)
# plot!(sol.t, getindex.(sol.u, 3), linewidth=2, label=L"\hat{m}", legendfontsize=15)
# plot!(sol.t, getindex.(sol.u, 4), linewidth=2, label=L"\hat{b}", legendfontsize=15)
# plot!(sol.t, getindex.(sol.u, 5), linewidth=2, label=L"\hat{k}", legendfontsize=15, legend=:bottomright)
# # savefig(p, "adaptationrule1.pdf")
# # savefig(p, "adaptationrule1.png")

fig = figure("Lyapunov Function", figsize=(10,6))
fig.clear()
ax = fig.add_subplot(111)
ax.plot(sol.t, V.(sol.u, sol.t), linewidth=2, label=L"$V$")
ax.plot(sol.t[2:end], diff(V.(sol.u, sol.t))./diff(sol.t)[1], linewidth=2, label=L"$\dot{V}$ (numerical)")
ax.plot(sol.t, Vdot.(sol.u, sol.t), linewidth=2, linestyle="--", label=L"$\dot{V}$")
ax.plot([0,tend], [0,0], linewidth=1, linestyle="dotted", color="k")
ax.tick_params(axis="both", which="major", labelsize=15)
ax.tick_params(axis="both", which="minor", labelsize=12)
ax.legend(fontsize=20)

fig2 = figure("State and Parameter Errors", figsize=(10,6))
fig2.clear()
ax2 = fig2.add_subplot(111)
ax2.plot(sol.t, xtilde.(sol.u, sol.t), label=L"\tilde{x}")
ax2.plot(sol.t, xtildedot.(sol.u, sol.t), label=L"\dot{\tilde{x}}")
ax2.plot(sol.t, m.-getindex.(sol.u, 3), label=L"\tilde{m}")
ax2.plot(sol.t, b.-getindex.(sol.u, 4), label=L"\tilde{b}")
ax2.plot(sol.t, k.-getindex.(sol.u, 5), label=L"\tilde{k}")
ax2.plot([0,tend], [0,0], linewidth=1, linestyle="dotted", color="k")
ax2.tick_params(axis="both", which="major", labelsize=15)
ax2.tick_params(axis="both", which="minor", labelsize=12)
ax2.legend(fontsize=20)

# Not sure how this should work at the moment.

F(t) = [0 0 0 λ/Γ[1,1]*xrefddot(t) 1/Γ[1,1]*xrefddot(t);
        0 0 0 λ/Γ[2,2]*xrefdot(t) 1/Γ[2,2]*xrefdot(t);
        0 0 0 λ/Γ[3,3]*xref(t) 1/Γ[3,3]*xref(t);
        0 0 0 1 0;
        -1/m*xrefddot(t) -1/m*xrefdot(t) -1/m*xref(t) -λ/m*(b+c) -(λ+(b+c)/m)]

# F(t) = [0 0 0 λ/Γ[1,1]*xrefddot(t) (1/Γ[1,1]-1/m)*xrefddot(t);
#         0 0 0 λ/Γ[2,2]*xrefdot(t) (1/Γ[2,2]-1/m)*xrefdot(t);
#         0 0 0 λ/Γ[3,3]*xref(t) (1/Γ[3,3]-1/m)*xref(t);
#         λ/Γ[1,1]*xrefddot(t) λ/Γ[2,2]*xrefdot(t) λ/Γ[3,3]*xref(t) 1 -λ/m*(b+c);
#         (1/Γ[1,1]-1/m)*xrefddot(t) (1/Γ[2,2]-1/m)*xrefdot(t) (1/Γ[3,3]-1/m)*xref(t) -λ/m*(b+c) -(λ+(b+c)/m)]

M(t) = F(t) + F(t)'
