""" Active Pressure Regulator Fluid System Classes """
__author__ = "Max Finkl"
__email__ = "max.finkl@warr.de"
__version__ = "0.1"

import numpy as np
import CoolProp.CoolProp as CP
import fluids

CP_P_MIN = 1e2
CP_P_MAX = 5e7

class FluidState:
    """
    Generic Fluid State Class
    used tp get information on the fluid properties at a lump
    """
    def __init__(self, system: object, component: object, pressure: float = 1e5, fluid: str = None, enthalpy: float = None, **kwargs):
        self.system = system
        self.component = component
        self.__pressure = pressure
        self.__enthalpy = None
        self.enthalpy2 = None
        if fluid is None:
            self.fluid = self.component.fluid
        else:
            self.fluid = fluid

    @property
    def pressure(self):
        return np.clip(self.__pressure, CP_P_MIN, CP_P_MAX)

    @pressure.setter
    def pressure(self, value):
        self.__pressure = value

    @property
    def enthalpy(self):
        if self.__enthalpy is None:
            if isinstance(self.component.upstream, Boundary) or self.component.upstream.fluid != self.fluid:
                self.__enthalpy = CP.PropsSI('H', 'T', 300, 'P', np.clip(self.pressure, CP_P_MIN, CP_P_MAX), self.fluid)
            else:
                self.__enthalpy = self.component.upstream.outlet.enthalpy
        return self.__enthalpy

    @enthalpy.setter
    def enthalpy(self, value):
        self.__enthalpy = value

    def density(self) -> float:
        return CP.PropsSI('D', 'H', self.enthalpy, 'P', np.clip(self.pressure, CP_P_MIN, CP_P_MAX), self.fluid) # CP.PropsSI('D', 'T', self.system.temperature_ambient, 'P', self.pressure, self.fluid) #

    def heat_capacity_ratio(self) -> float:
        """
        heat capacity ratio or isentropic_expansion_coefficient
        :return:
        """
        x = None#CP.PropsSI('isentropic_expansion_coefficient', 'H', self.enthalpy, 'P', np.clip(self.pressure, CP_P_MIN, CP_P_MAX), self.fluid) # CP.PropsSI('isentropic_expansion_coefficient', 'T', self.system.temperature_ambient, 'P', self.pressure, self.fluid) ##None
        h_init = self.enthalpy
        if self.enthalpy2 is None:
            self.enthalpy2 = self.enthalpy
        while x is None:
            try:
                x = CP.PropsSI('isentropic_expansion_coefficient', 'H', self.enthalpy2, 'P', self.pressure, self.fluid) # CP.PropsSI('isentropic_expansion_coefficient', 'T', self.system.temperature_ambient, 'P', self.pressure, self.fluid) #
            except:
                self.enthalpy2 = 0#self.enthalpy2 + np.abs(h_init) * 1e-2
                #print(str(self.__enthalpy))
        return x 

    def speed_of_sound(self) -> float:
        return CP.PropsSI('A', 'H', self.enthalpy, 'P', np.clip(self.pressure, CP_P_MIN, CP_P_MAX), self.fluid) # CP.PropsSI('A', 'T', self.system.temperature_ambient, 'P', self.pressure, self.fluid) #

    def bulk_modulus(self) -> float:
        #return self.speed_of_sound()**2 * self.density()
        return np.clip(self.pressure * self.heat_capacity_ratio(), 0, 1e100)  # alternative: self.speed_of_sound() ** 2 * self.density()

    def viscosity_dyn(self) -> float:
        return CP.PropsSI('V', 'H', self.enthalpy, 'P', np.clip(self.pressure, CP_P_MIN, CP_P_MAX), self.fluid) # CP.PropsSI('V', 'T', self.system.temperature_ambient, 'P', self.pressure, self.fluid) #

    def temperature(self) -> float:
        return CP.PropsSI('T', 'H', self.enthalpy, 'P', np.clip(self.pressure, CP_P_MIN, CP_P_MAX), self.fluid)

class Boundary():
    """
    System boundary component for setting mass flow leaving or entering the system
    """
    def __init__(self, source: object = None, m_dot: float = 0, pressure: float = 1e5, **kwargs):
        super().__init__(**kwargs)
        self.m_dot = m_dot

class FluidSystem:
    """
    Fluid system
    used to set properties for all components that get passed this fluid system
    """
    def __init__(self, temperature_ambient: float = 293, enthalpy_initial: float = None, **kwargs):
        self.temperature_ambient = temperature_ambient
        self.enthalpy_initial = enthalpy_initial

class FluidComponent:
    """
    Fluid System Component
    used to house component information like interfacing components
    """
    def __init__(self, fluid: str, system: FluidSystem = None, name: str = 'unnamed', upstream: object = None, **kwargs):
        self.fluid = fluid
        self.system = system
        self.name = name
        self.upstream = upstream
        self.downstream = Boundary()
        if upstream is None:
            self.upstream = Boundary()
        else: self.upstream.downstream = self

class FlowComponent(FluidComponent):
    """
    Generic Flow Component
    used to store mass flow lumps
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inlet = Inlet(component=self, **kwargs)
        self.outlet = Outlet(component=self, **kwargs)
        self.__m_dot = 0
        self.__velocity = 0

    @property
    def m_dot(self):
        return self.__m_dot

    @m_dot.setter
    def m_dot(self, value):
        self.__m_dot = value

class Inlet(FluidState):
    """
    Inlet class
    used as a component interface to handle interface lump linking
    """
    def __init__(self, pressure: float = 1e5, **kwargs):
        super().__init__(**kwargs)
        self.__pressure = pressure

    @property
    def pressure(self):
        if self.component.upstream is not None:
            try:
                self.__pressure = self.component.upstream.outlet.pressure
            except:
                # print('no outlet found at:', self.component.upstream.name)
                pass
        return np.clip(self.__pressure, CP_P_MIN, CP_P_MAX)

    @pressure.setter
    def pressure(self, value):
        self.__pressure = value

class Outlet(FluidState):
    """
    Outlet class
    used as a component interface to handle interface lump linking
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Pipe(FlowComponent):
    """
    Pipe (lumped parameter model)
    used to store one inlet and outlet and mass flow lump each
    """
    def __init__(self, radius: float, length: float, roughness: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        self.length = length
        self.roughness = roughness

    def m_2dot(self) -> float:
        """
        change of mass flow in the center of the pipe over time
        :return: differential equation for the solver
        """
        return self.area() / self.length * (self.inlet.pressure - self.outlet.pressure - self.friction() * self.m_dot * abs(self.m_dot) / self.inlet.density())

    def p_dot_out(self) -> float:
        """
        change of pressure at the pipe outlet over time
        :return: differential equation for the solver
        """
        return (self.m_dot - self.downstream.m_dot) * np.sqrt(self.outlet.bulk_modulus() / self.outlet.density())**2 / self.volume()

    def p_dot_in(self) -> float:
        """
        change of pressure at the pipe inlet over time
        :return: differential equation for the solver
        """
        return (self.upstream.m_dot - self.m_dot) * np.sqrt(self.inlet.bulk_modulus() / self.inlet.density())**2 / self.volume()

    def hydraulic_diameter(self) -> float:
        return 4 * self.area() / self.circumference()

    def roughness_ratio(self) -> float:
        return self.roughness / self.hydraulic_diameter()

    def friction_coefficient(self) -> float:
        return fluids.friction.friction_factor(self.reynolds(), eD=self.roughness_ratio(), Method='Brkic_2011_2', Darcy=True)

    def friction(self) -> float: 
        return self.friction_coefficient() * self.length / (self.radius * 4 * self.area() ** 2) # (0.316 * self.length / (2 * self.radius * self.reynolds() ** 0.25))/ (2 * self.area() ** 2)

    def reynolds(self) -> float:
        return self.inlet.density() * (self.velocity() + 1e-9) * self.hydraulic_diameter() / self.inlet.viscosity_dyn()

    def velocity(self) -> float:
        return np.sqrt(self.m_dot ** 2) / self.inlet.density() / self.area()

    def circumference(self) -> float:
        return 2 * self.radius * np.pi

    def area(self) -> float:
        return self.radius ** 2 * np.pi

    def volume(self) -> float:
        return self.area() * self.length

class Tank(FlowComponent, FluidState):
    """
    Tank (lumped parameter model)
    used for single fluid tank
    """
    def __init__(self, volume: float, **kwargs):
        FlowComponent.__init__(self, **kwargs)
        FluidState.__init__(self, component=self, **kwargs)
        self.volume = volume

    @property
    def m_dot(self):
        return self.upstream.m_dot - self.downstream.m_dot

    @m_dot.setter
    def m_dot(self, value):
        pass

    @property
    def pressure(self):
        if self.component.upstream is not None:
            try: # only executed if outlet is present
                self.__pressure = self.component.upstream.outlet.pressure
                self.inlet.pressure = self.__pressure
                self.outlet.pressure = self.__pressure
            except:
                pass
        return self.__pressure

    @pressure.setter
    def pressure(self, value):
        self.__pressure = value
        self.inlet.pressure = value
        self.outlet.pressure = value
        pass

    def p_dot(self) -> float:
        """
        change of pressure in the tank over time
        :return: differential equation for the solver
        """
        return (self.upstream.m_dot - self.downstream.m_dot) * np.sqrt(self.inlet.bulk_modulus() / self.inlet.density()) ** 2 / self.volume

class Valve(FlowComponent):
    """
    Valve (lumped parameter model)
    """

    def __init__(self, radius_bore: float, radius_port: float, angle: float = 0, pt1_time_constant_opening: float = 0.1, pt1_time_constant_closing: float = None, Cd: float = 0.64, **kwargs):
        super().__init__(**kwargs)
        self.radius_bore = radius_bore
        self.radius_port = radius_port
        self.__angle = angle
        self.angle_command = angle
        self.pt1_time_constant_opening = pt1_time_constant_opening
        if pt1_time_constant_closing is None:
            self.pt1_time_constant_closing = pt1_time_constant_opening
        else:
            self.pt1_time_constant_closing = pt1_time_constant_closing
        self.angle_max = 90
        self.angle_min = 0
        self.Cd = Cd

    @property
    def angle(self):
        return np.clip(self.__angle, self.angle_min, self.angle_max)

    @angle.setter
    def angle(self, value):
        self.__angle = value

    def m_dot_update(self):
        # valve area equation -> source: https://de.mathworks.com/help/hydro/ref/ballvalveil.html?searchHighlight=E&s_tid=doc_srchtitle
        lambda_bore = (self.radius_bore ** 2 - self.radius_port ** 2) / (2 * self.radius_bore)
        lambda_port = (self.radius_port ** 2 - self.radius_bore ** 2) / (2 * self.radius_port)

        A_valve = np.sin(np.deg2rad(self.angle)) * self.radius_bore ** 2 * (np.arccos(lambda_bore) - lambda_bore * np.sqrt(1 - lambda_bore) + self.radius_port ** 2 * (np.arccos(lambda_port) - lambda_port * np.sqrt(1 - lambda_port)))
        # mass flow calculation
        Cd = self.Cd # discharge coefficient ball valve
        p_in = self.inlet.pressure
        p_out = self.outlet.pressure
        gamma = self.inlet.heat_capacity_ratio()
        rho = self.inlet.density()
        visc = self.inlet.viscosity_dyn()
        p_ratio = p_out / p_in
        p_ratio_choke = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        if p_ratio < p_ratio_choke: # choked case
            # mass flow equation -> source: https://en.wikipedia.org/wiki/Choked_flow
            m_dot = Cd * A_valve * np.sqrt(gamma * rho * p_in * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)))
        else: # unchoked case
            # mass flow equation -> source: https://de.mathworks.com/help/hydro/ref/ballvalveil.html?searchHighlight=E&s_tid=doc_srchtitle
            Reynolds_crit = 150
            dp = p_in - p_out
            dp_crit = np.pi * rho / (8 * A_valve + 1e-9) * (visc * Reynolds_crit / Cd) ** 2
            m_dot = Cd * A_valve * np.sqrt(2 * rho) * dp / (dp ** 2 + dp_crit ** 2) ** 0.25
        self.m_dot = m_dot

        #self.m_dot = Cd * A_valve * np.sqrt(2 * rho * (p_in - p_out))

    def angle_dot(self) -> float:
        angle_command_deg = (self.angle_command * self.angle_max)
        if angle_command_deg >= self.angle:
            angle_dot = (angle_command_deg - self.angle) / self.pt1_time_constant_opening
        else:
            angle_dot = (angle_command_deg - self.angle) / self.pt1_time_constant_closing
        return angle_dot

class PropellantTank(Tank):
    """
    Propellant Tank Class
    used for propellant fill ratio and ullage interaction
    """
    def __init__(self, propellant: str, fill_ratio: float = 0.65, **kwargs):
        super().__init__(**kwargs)
        self.propellant = PropellantState(component=self, fluid=propellant, volume=self.volume * fill_ratio)
        self.__m_dot = 0
        self.__velocity = 0
        self.__fill_ratio = fill_ratio

    @property
    def fill_ratio(self):
        self.__fill_ratio = self.propellant.volume / self.volume
        return self.__fill_ratio

    @fill_ratio.setter
    def fill_ratio(self, value):
        self.__fill_ratio = value

    def pressurant_volume(self):
        return self.volume * (1 - self.fill_ratio)

    def pressure_update(self, volume_propellant_new):
        m_pressurant = self.pressurant_volume() * self.inlet.density()
        self.propellant.volume = volume_propellant_new
        density_pressurant_new = m_pressurant / self.pressurant_volume()
        self.inlet.pressure = CP.PropsSI('P', 'H', self.inlet.enthalpy, 'D', density_pressurant_new, self.fluid)
        return self.inlet.pressure

    def v_dot(self):
        return - self.downstream.m_dot / self.propellant.density()

    def p_dot(self) -> float:
        return (self.upstream.m_dot + self.v_dot() * self.inlet.density()) * np.sqrt(self.inlet.bulk_modulus() / self.inlet.density()) ** 2 / self.pressurant_volume()

class PropellantState(FluidState):
    """
    Propellant State Class
    used for propellant volume storage
    """
    def __init__(self, volume=None, **kwargs):
        super().__init__(**kwargs, system=None)
        self.__volume = volume

    @property
    def volume(self):
        return self.__volume

    @volume.setter
    def volume(self, value):
        self.__volume = value



class TestClass:
    def __init__(self, **kwargs):
        self.topo()
        pass

    def topo(self):
            fluid = 'Propane'
            self.system = FluidSystem(fluid=fluid)
            self.cavity_upstream = Tank(system=self.system, fluid=fluid, volume=100)
            self.pipe_upstream = Pipe(system=self.system, fluid=fluid, upstream=self.cavity_upstream, radius=0.025, length=1)
            self.valve = Valve(system=self.system, fluid=fluid, upstream=self.pipe_upstream, radius_bore=0.025, radius_port=0.025, pt1_time_constant_opening=0.1, pt1_time_constant_closing=0.05, Cd=0.75)
            self.pipe_downstream = Pipe(system=self.system, fluid=fluid, upstream=self.valve, radius=0.025, length=1)
            self.cavity_downstream = Tank(system=self.system, fluid=fluid, upstream=self.pipe_downstream, volume=0.001)
            print('Plant topology created')
            pass

    def ode(self, t, x):
        ## Inputs
        self.valve.angle_command = 0 if t < 0.1 else 1

        ## States
        self.cavity_upstream.pressure = x[0]
        self.pipe_upstream.m_dot = x[1]
        self.pipe_upstream.outlet.pressure = x[2]
        self.valve.angle = x[3]
        self.valve.outlet.pressure = x[4]
        self.pipe_downstream.m_dot = x[5]
        self.pipe_downstream.outlet.pressure = x[6]

        ## DAE
        self.valve.m_dot_update()

        ## ODE
        ode0 = self.cavity_upstream.p_dot()
        ode1 = self.pipe_upstream.m_2dot()
        ode2 = self.pipe_upstream.p_dot_out()
        ode3 = self.valve.angle_dot()
        ode4 = self.pipe_downstream.p_dot_in()
        ode5 = self.pipe_downstream.m_2dot()
        ode6 = self.cavity_downstream.p_dot()

        return np.array([ode0, ode1, ode2, ode3, ode4, ode5, ode6])

x = TestClass()

#y = CP.PropsSI('Q', 'T', 300, 'P', 1e5, 'Propane')
import scipy
result = scipy.integrate.solve_ivp(x.ode, t_span=(0, 0.5), y0=np.array([6e5, 0, 6e5, 0, 5e5, 0, 5e5]), method='RK45')




# def plot_propane_enthalpy_isobars(T_min=90, T_max=300, pressures=None, n_points=100):
#     """
#     Plot CoolProp data for propane enthalpy between T_min and T_max with isobars
    
#     Args:
#         T_min: Minimum temperature (K), default 90
#         T_max: Maximum temperature (K), default 300
#         pressures: List of pressures (Pa) for isobars, default [1e5, 5e5, 10e5, 20e5, 30e5]
#         n_points: Number of points for each isobar
#     """
#     if pressures is None:
#         pressures = [1e5, 5e5, 10e5, 20e5, 30e5]  # Default pressures in Pa
    
#     fluid = 'Propane'
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Plot isobars
#     for P in pressures:
#         T_range = np.linspace(T_min, T_max, n_points)
#         H_range = []
        
#         for T in T_range:
#             try:
#                 H = CP.PropsSI('D', 'T', T, 'P', P, fluid)
#                 H_range.append(H)  # Convert to kJ/kg
#             except:
#                 H_range.append(np.nan)
        
#         ax.plot(T_range, H_range, label=f'{P/1e5:.1f} bar', linewidth=2)
    
#     ax.set_xlabel('Temperature (K)', fontsize=12)
#     ax.set_ylabel('Enthalpy (kJ/kg)', fontsize=12)
#     ax.set_title('Propane Enthalpy vs Temperature - Isobars', fontsize=14)
#     ax.grid(True, alpha=0.3)
#     ax.legend(title='Pressure', fontsize=10)
    
#     plt.tight_layout()
#     return fig, ax
# fig, ax = plot_propane_enthalpy_isobars(pressures=[1e2, 1e5, 6e5, 100e5, 300e5, 500e5])
# plt.show()


import matplotlib.pyplot as plt
plt.plot(result.t, result.y[0], label='P1')
plt.plot(result.t, result.y[1], linestyle='--', label='m1')
plt.plot(result.t, result.y[2], label='P2')
plt.plot(result.t, result.y[3] / 90 * 6e5, label='v')
plt.plot(result.t, result.y[4], label='P3')
plt.plot(result.t, result.y[5], linestyle='--', label='m2')
plt.plot(result.t, result.y[6], label='P4')
plt.legend()
plt.show()
a