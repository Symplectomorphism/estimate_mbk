using OrdinaryDiffEq
using Plots
using LaTeXStrings

m, b = (0.7, 0.5)
A0, ω, φ = (0.5, 2*π*0.5, 60*π/180)
k, λ = (1, 1)

xref(t) = A0*sin(ω*t + φ)
xrefdot(t) = A0*ω*cos(ω*t + φ)
xrefddot(t) = -A0*ω*ω*sin(ω*t + φ)

xtilde(x, t) = x[1] - xref(t)
xtildedot(x, t) = x[2] - xrefdot(t)

r(x, t) = xtildedot(x, t) + λ*xtilde(x, t)
u(x, t) = x[3]*x[2] + m*(xrefddot(t) - λ*xtildedot(x, t)) - k*r(x, t)
# u(x, t) = x[3]*x[2] + m*(xrefddot(t) - λ*xtildedot(x, t)) - m*k*r(x, t)
f(x, t) = -r(x, t)*x[2]

function eom!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = 1/m*(u(x, t) - b*x[2])
    dx[3] = (x[3] > 0 ? f(x,t) : 1)
    # dx[3] = f(x,t)
end

x0 = [0, 0, 0.0]
tspan = (0, 20.0)
prob = ODEProblem(eom!, x0, tspan, saveat=range(tspan[1]; stop=tspan[2], length=1001))
sol = solve(prob, Tsit5())

x = [getindex.(sol.u, 1), getindex.(sol.u, 2), getindex.(sol.u, 3)]

# p = plot(sol.t, xtilde.(sol.u, sol.t), linewidth=2, label=L"\tilde{x}", legendfontsize=15)
# plot!(sol.t, xtildedot.(sol.u, sol.t), linewidth=2, label=L"\frac{d\tilde{x}}{dt}", legendfontsize=15)
p = plot(sol.t, getindex.(sol.u, 1), linewidth=2, label=L"\tilde{x}", legendfontsize=15)
plot!(sol.t, getindex.(sol.u, 1), linewidth=2, label=L"\frac{d\tilde{x}}{dt}", legendfontsize=15)
plot!(sol.t, getindex.(sol.u, 3), linewidth=2, label=L"\hat{b}", legendfontsize=15)
savefig(p, "adaptationrule1.svg")

p = plot(sol.t, xtilde.(sol.u, sol.t), linewidth=2, label=L"\tilde{x}", legendfontsize=15)
plot!(sol.t, xtildedot.(sol.u, sol.t), linewidth=2, label=L"\frac{d\tilde{x}}{dt}", legendfontsize=15)
plot!(sol.t, b.-getindex.(sol.u, 3), linewidth=2, label=L"\tilde{b}", legendfontsize=15)
savefig(p, "adaptationrule2.svg")
