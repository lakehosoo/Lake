from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.simulator import Simulator

def create_model():
    m = ConcreteModel()

    m.t = ContinuousSet(bounds=(0,1))
    m.a = Var(m.t)
    m.b = Var(m.t)
    m.c = Var(m.t)
    m.k1 = Param(initialize=5)
    m.k2 = Param(initialize=1)

    m.dadt = DerivativeVar(m.a, wrt=m.t)
    m.dbdt = DerivativeVar(m.b, wrt=m.t)
    m.dcdt = DerivativeVar(m.c, wrt=m.t)

    m.a[0].fix(1)
    m.b[0].fix(0)
    m.c[0].fix(0)

    def _dadt(m,i):
        return m.dadt[i] == -m.k1*m.a[i]
    m.dadtcon = Constraint(m.t, rule=_dadt)

    def _dbdt(m,i):
        return m.dbdt[i] == m.k1*m.a[i] - m.k2*m.b[i]
    m.dbdtcon = Constraint(m.t, rule=_dbdt)

    def _dcdt(m,i):
        return m.dcdt[i] == m.k2*m.b[i]
    m.dcdtcon = Constraint(m.t, rule=_dcdt)

    return m


def simulate_model(m):
    # Simulate the model using scipy
    sim = Simulator(m, package='scipy')
    tsim, profiles = sim.simulate(numpoints=100, integrator='vode')

    # Discretize model using Orthogonal Collocation
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=10, ncp=5)

    # Initialize the discretized model using the simulator profiles
    sim.initialize_model()

    return sim, tsim, profiles


def plot_result(m, sim, tsim, profiles):
    import matplotlib.pyplot as plt

    time = list(m.t)
    a = [value(m.a[t]) for t in m.t]
    b = [value(m.b[t]) for t in m.t]
    c = [value(m.c[t]) for t in m.t]

    varorder = sim.get_variable_order()

    for idx, v in enumerate(varorder):
        plt.plot(tsim, profiles[:, idx], label=v)
    plt.plot(time, a, 'o', label='a interp')
    plt.plot(time, b, 'o', label='b interp')
    plt.plot(time, c, 'o', label='c interp')
    plt.xlabel('t')
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    model = create_model()
    sim, tsim, profiles = simulate_model(model)
    plot_result(model, sim, tsim, profiles)
