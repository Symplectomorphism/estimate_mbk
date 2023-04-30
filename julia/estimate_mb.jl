using OrdinaryDiffEq
using Plots
using LaTeXStrings
using MAT

file = matopen("./hardware/estimation_station3_new.mat")
data = read(file, "data")
close(file)

m, b = (data[4,end], data[5, end])
A0, ω, φ = (0.3, 2*π*0.3, 0*π/180)
k, λ = (1, 4)

xref(t) = A0*sin(ω*t + φ)
xrefdot(t) = A0*ω*cos(ω*t + φ)
xrefddot(t) = -A0*ω*ω*sin(ω*t + φ)

xtilde(x, t) = x[1] - xref(t)
xtildedot(x, t) = x[2] - xrefdot(t)

r(x, t) = xtildedot(x, t) + λ*xtilde(x, t)
u(x, t) = x[3]*x[2] + x[4]*(xrefddot(t) - λ*xtildedot(x, t)) - k*r(x, t)
# u(x, t) = x[3]*x[2] + m*(xrefddot(t) - λ*xtildedot(x, t)) - m*k*r(x, t)
f(x, t) = -r(x, t)*x[2]
g(x, t) = -r(x, t)*(xrefddot(t) - λ*xtildedot(x, t))

function eom!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = 1/m*(u(x, t) - b*x[2])
    # dx[3] = (x[3] > 0 ? f(x,t) : 1)
    # dx[4] = (x[4] > 0 ? g(x,t) : 1)
    dx[3] = f(x,t)
    dx[4] = g(x,t)
end

x0 = [0, 0, -0.25, -0.5]
tspan = (0, data[1,end])
prob = ODEProblem(eom!, x0, tspan, saveat=range(tspan[1]; stop=tspan[2], length=10001))
sol = solve(prob, Tsit5())

x = [getindex.(sol.u, 1), getindex.(sol.u, 2), getindex.(sol.u, 3), getindex.(sol.u, 4)]

p = plot(sol.t, xtilde.(sol.u, sol.t), linewidth=2, label=L"\tilde{x}", legendfontsize=15)
plot!(sol.t, xtildedot.(sol.u, sol.t), linewidth=2, label=L"\frac{d\tilde{x}}{dt}", legendfontsize=15)
# p = plot(sol.t, getindex.(sol.u, 1), linewidth=2, label=L"x", legendfontsize=15)
# plot!(sol.t, getindex.(sol.u, 1), linewidth=2, label=L"\frac{dx}{dt}", legendfontsize=15)
plot!(sol.t, getindex.(sol.u, 4), linewidth=2, label=L"\hat{m}", legendfontsize=15)
plot!(sol.t, getindex.(sol.u, 3), linewidth=2, label=L"\hat{b}", legendfontsize=15, legend=:bottomright, )
# savefig(p, "adaptationrule1.pdf")
# savefig(p, "adaptationrule1.png")

p = plot(sol.t, xtilde.(sol.u, sol.t), linewidth=2, label=L"\tilde{x}", legendfontsize=15)
plot!(sol.t, xtildedot.(sol.u, sol.t), linewidth=2, label=L"\frac{d\tilde{x}}{dt}", legendfontsize=15)
plot!(sol.t, m.-getindex.(sol.u, 4), linewidth=2, label=L"\tilde{m}", legendfontsize=15)
plot!(sol.t, b.-getindex.(sol.u, 3), linewidth=2, label=L"\tilde{b}", legendfontsize=15)
# savefig(p, "adaptationrule2.pdf")
# savefig(p, "adaptationrule2.png")


p = plot(sol.t, xtilde.(sol.u, sol.t), linewidth=2, label=L"\tilde{x}: sim", legendfontsize=15)
plot!(data[1,:], data[2,:], linewidth=2, label=L"\tilde{x}: real", linestyle=:dash, legendfontsize=15)
savefig(p, "./TeX/figures/xtilde_real_sim.pdf")

p = plot(sol.t, xtildedot.(sol.u, sol.t), linewidth=2, label=L"\frac{d\tilde{x}}{dt}: sim", legendfontsize=15)
plot!(data[1,:], data[3,:], linewidth=2, label=L"\frac{d\tilde{x}}{dt}: real", linestyle=:dash, legendfontsize=15)
savefig(p, "./TeX/figures/xtildedot_real_sim.pdf")

p = plot(sol.t, getindex.(sol.u, 4), linewidth=2, label=L"\hat{m}: sim", legendfontsize=15)
plot!(data[1,:], data[4,:], linewidth=2, label=L"\hat{m}: real", linestyle=:dash, legendfontsize=15)
savefig(p, "./TeX/figures/mhat_real_sim.pdf")

p = plot(sol.t, getindex.(sol.u, 3), linewidth=2, label=L"\hat{b}: sim", legendfontsize=15)
plot!(data[1,:], data[5,:], linewidth=2, label=L"\hat{b}: real", linestyle=:dash, legendfontsize=15)
savefig(p, "./TeX/figures/bhat_real_sim.pdf")
