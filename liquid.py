from operator import length_hint
import numpy as np
import CoolProp.CoolProp as CP
import fluids

# Define pressure limits for CoolProp calculations to avoid errors
CP_P_MIN = 1e2
CP_P_MAX = 1e8

class FluidElement():
    def __init__(self) -> None:
        pass
    
    def inductance(self, density, length, area):
        return density * length / area

    def resistance(self, friction_coefficient, velocity, density, length, hydraulic_diameter):
        signed_square_velocity = velocity * np.abs(velocity)
        return friction_coefficient * length * density * signed_square_velocity /(2 * hydraulic_diameter)
    
    def capacitance(self, speed_of_sound, density, volume):
        bulk_modulus = speed_of_sound ** 2 * density
        return volume / bulk_modulus

class FluidODE():
    def __init__(self):
        self.FLE = FluidElement()
        
    def m_dot2(self, delta_p, friction_coefficient, velocity, density, length, hydraulic_diameter, area):
        pressure_difference = delta_p - self.FLE.resistance(friction_coefficient, velocity, density, length, hydraulic_diameter)
        return pressure_difference / self.FLE.inductance(density, length, area)

    def p_dot(self, delta_m_dot, density, speed_of_sound, volume):
        return delta_m_dot / density / self.FLE.capacitance(speed_of_sound, density, volume)

class RobustCoolProp:
    """
    Wrapper for coolprop to prevent solver issues
    Truncates inputs with the specified limits
    """
    def __init__(self, fluid, enforce_phase: str = None, **kwargs):
        self.fluid = fluid
        if enforce_phase is None:
            self.enforce_phase = enforce_phase
        else:
            self.enforce_phase = CP.PhaseSI('H', self.enthalpy, 'P', self.pressure, self.fluid)
        # CONSTANTS
        self.PRESSURE_LIMIT = [1e2, 1e8]
        self.TEMPERATURE_LIMIT = [1e1, 5e3]
        self.BULK_MODULUS_LIMIT = [0, 1e100]

    def pressure_robust(self, pressure):
        return np.clip(pressure, self.PRESSURE_LIMIT[0], self.PRESSURE_LIMIT[1])

    def temperature_robust(self, temperature):
        return np.clip(temperature, self.TEMPERATURE_LIMIT[0], self.TEMPERATURE_LIMIT[1])

    def enthalpy(self, temperature, pressure):
        string = 'T'
        if self.enforce_phase is not None:
            string = string + '|' + self.enforce_phase
        return CP.PropsSI('H', string, temperature, 'P', self.pressure_robust(pressure), self.fluid)

    def density(self, enthalpy, pressure) -> float:
        string = 'H'
        if self.enforce_phase is not None:
            string = string + '|' + self.enforce_phase
        return CP.PropsSI('D', string, enthalpy, 'P', self.pressure_robust(pressure), self.fluid)

    def speed_of_sound(self, enthalpy, pressure) -> float:
        string = 'H'
        if self.enforce_phase is not None:
            string = string + '|' + self.enforce_phase
        return CP.PropsSI('A', string, enthalpy, 'P', self.pressure_robust(pressure), self.fluid)

    def viscosity_dyn(self, enthalpy, pressure) -> float:
        string = 'H'
        if self.enforce_phase is not None:
            string = string + '|' + self.enforce_phase
        return CP.PropsSI('V', string, enthalpy, 'P', self.pressure_robust(pressure), self.fluid)

    def temperature(self, enthalpy, pressure) -> float:
        string = 'H'
        if self.enforce_phase is not None:
            string = string + '|' + self.enforce_phase
        return CP.PropsSI('T', string, enthalpy, 'P', self.pressure_robust(pressure), self.fluid)

    # def heat_capacity_ratio(self, enthalpy, pressure) -> float:
    #     string = 'H'
    #     if self.enforce_phase is not None:
    #         string = string + '|' + self.enforce_phase
    #     return CP.PropsSI('isentropic_expansion_coefficient', string, enthalpy, 'P', self.pressure_robust(pressure), self.fluid)

    # def bulk_modulus(self, enthalpy, pressure) -> float:
    #     # self.speed_of_sound() ** 2 * self.density()
    #     return self.pressure_robust(pressure) * self.heat_capacity_ratio(enthalpy, pressure)

    def phase(self, enthalpy, temperature) -> str:
        return CP.PhaseSI('H', enthalpy, 'T', self.temperature_robust(temperature), self.fluid)

class System:
    """
    Fluid system
    Used to set properties for all components that get passed this fluid system.
    """
    def __init__(self, fluid: str = 'Nitrogen', temperature_init: float = 293, components: list = None, **kwargs):
        self.fluid = fluid
        self.temperature_init = temperature_init
        self.coolprop = RobustCoolProp(fluid = self.fluid)
        self.__components = components
        if self.__components is not None:
            self.components_init()
        #self.robust_coolprop.enforce_phase = self.robust_coolprop.phase(enthalpy = self.enthalpy_init(), temperature = temperature_init)
        
    @property
    def components(self):
        return self.__components

    @components.setter
    def components(self, payload):
        self.__components = payload
        self.components_init()

    def components_init(self):
        for id, component in enumerate(self.components):
            component.system = self
            component.fluid = self.fluid
            component.enthalpy = self.enthalpy_init()
            if id > 0:
                component.upstream_component = self.components[id-1]

    def enthalpy_init(self):
        pressure = 1e5
        for component in self.components:
            p = 0
            try:
                p = component.pressure
            except:
                p = component.node[-1].pressure
            if p > pressure:
                pressure = p
        return self.coolprop.enthalpy(temperature = self.temperature_init, pressure = pressure)

    def ode(self, t, x):
        ## Inputs
        self.components[2].angle_command = 0 if t < 0.1 else 1

        ## States
        self.components[0].node[-1].pressure = x[0]
        self.components[1].m_dot = x[1]
        self.components[1].node[-1].pressure = x[2]
        self.components[2].angle = x[3]
        self.components[2].node[-1].pressure = x[4]
        self.components[3].m_dot = x[5]
        self.components[3].node[-1].pressure = x[6]

        ## ODE
        ode0 = self.components[0].node[-1].p_dot()
        ode1 = self.components[1].m_dot2()
        ode2 = self.components[1].node[-1].p_dot()
        ode3 = self.components[2].angle_dot()
        ode4 = self.components[3].node[-1].p_dot()
        ode5 = self.components[3].m_dot2()
        ode6 = self.components[4].node[-1].p_dot()
        print(t, x)
        return np.array([ode0, ode1, ode2, ode3, ode4, ode5, ode6])

class Component:
    """
    Component to be initialized as part of a system
    Used to house component information like interfacing components.
    """
    def __init__(self, name: str = 'unnamed', system: System = None, upstream_component: object = None, **kwargs):
        self.name = name
        self.__system = system
        self.__upstream_component = upstream_component
        self.downstream_component = None
    
    @property
    def system(self):
        if self.__system is None:
            raise BaseException('A system needs to be assigned to component "' + self.name + '"')
        else:
            return self.__system

    @system.setter
    def system(self, payload):
        self.__system = payload

    @property
    def upstream_component(self):
        return self.__upstream_component

    @upstream_component.setter
    def upstream_component(self, value):
        self.__upstream_component = value
        if self.__upstream_component is not None:
            self.__upstream_component.downstream_component = self

class NodeLump:
    """
    Node lump handles properties at the end of link lumps
    """
    def __init__(self, component: object, pressure: float = 1e5, **kwargs):
        self.component = component
        self.ode = FluidODE()
        self.__pressure = pressure
        self.__enthalpy = None
        self.__coolprop = None

    def net_m_dot(self):
        if self.component.m_dot is None:
            m_dot_in = 0
        else:
            m_dot_in = self.component.m_dot
        if self.component.downstream_component is None:
            m_dot_out = 0
        elif self.component.downstream_component.m_dot is None:
            m_dot_out = 0
        else:
            m_dot_out = self.component.downstream_component.m_dot
        return m_dot_in - m_dot_out

    def net_volume(self):
        return self.component.volume

    def p_dot(self) -> float:
        return self.ode.p_dot(delta_m_dot=self.net_m_dot(), density=self.density(), speed_of_sound=self.speed_of_sound(), volume=self.net_volume())

    @property
    def coolprop(self):
        if self.__coolprop is None:
            self.__coolprop = self.component.system.coolprop
        return self.__coolprop

    @property
    def pressure(self):
        return self.__pressure

    @pressure.setter
    def pressure(self, value):
        self.__pressure = value

    @property
    def enthalpy(self):
        self.__enthalpy = self.component.enthalpy
        return self.__enthalpy

    @enthalpy.setter
    def enthalpy(self, value):
        self.__enthalpy = value

    def density(self) -> float:
        return self.coolprop.density(self.enthalpy, self.pressure)

    def heat_capacity_ratio(self) -> float:
        return self.coolprop.heat_capacity_ratio(self.enthalpy, self.pressure)

    def speed_of_sound(self) -> float:
        return self.coolprop.speed_of_sound(self.enthalpy, self.pressure)

    def bulk_modulus(self) -> float:
        return self.coolprop.bulk_modulus(self.enthalpy, self.pressure)

    def viscosity_dyn(self) -> float:
        return self.coolprop.viscosity_dyn(self.enthalpy, self.pressure)

    def temperature(self) -> float:
        return self.coolprop.temperature(self.enthalpy, self.pressure)

    def phase(self) -> str:
        return self.coolprop.phase(self.enthalpy, self.temperature())

class LinkLump:
    """
    Link lump handles properties at a link section between node lumps
    """
    def __init__(self, **kwargs) -> None:
        self.ode = FluidODE()
        self.__m_dot = 0

    def m_dot2(self):
        return 0 # skeleton overwritten by component

    @property
    def m_dot(self):
        return self.__m_dot

    @m_dot.setter
    def m_dot(self, value):
        self.__m_dot = value

class TwoPort(Component):
    """
    Generic component with two ports as inlet and outlet node lumps
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.node = [NodeLump(component=self, **kwargs), NodeLump(component=self, **kwargs)]

    def delta_pressure(self):
        return self.node[-1].pressure - self.node[0].pressure

class Cavity(Component):
    """
    Cavity (lumped parameter model), defined as a node lump
    """
    def __init__(self, volume: float, **kwargs):
        super().__init__(**kwargs)
        self.node = [NodeLump(component=self, **kwargs)]
        self.m_dot = None
        self.volume = volume

class Pipe(TwoPort, LinkLump):
    """
    Pipe (lumped parameter model)
    """
    def __init__(self, radius: float, length: float, roughness: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.ode = FluidODE()
        self.radius = radius
        self.length = length
        self.roughness = roughness

    def m_dot2(self):
        return self.ode.m_dot2(delta_p=self.delta_pressure(), friction_coefficient=self.friction_coefficient(), velocity=self.velocity(), 
            density=self.density(), length=self.length, hydraulic_diameter=self.hydraulic_diameter(), area=self.area())

    def area(self) -> float:
        return self.radius ** 2 * np.pi

    def circumference(self) -> float:
        return 2 * self.radius * np.pi

    def hydraulic_diameter(self) -> float:
        return 4 * self.area() / self.circumference()

    def roughness_ratio(self) -> float:
        return self.roughness / self.hydraulic_diameter()

    def friction_coefficient(self) -> float:
        return fluids.friction.friction_factor(self.reynolds(), eD=self.roughness_ratio(), Method='Brkic_2011_2', Darcy=True)

    def reynolds(self) -> float:
        return self.density() * (self.velocity() + 1e-9) * self.hydraulic_diameter() / self.viscosity_dyn()

    def velocity(self) -> float:
        return self.m_dot / (self.density() * self.area())

    @property
    def volume(self) -> float:
        return self.area() * self.length

    def density(self):
        return self.node[0].density()

    def viscosity_dyn(self):
        return self.node[0].viscosity_dyn()

class Valve(TwoPort, LinkLump):
    """
    Valve (lumped parameter model)
    Mass flow calculation is now phase-aware for liquids and gases.
    """
    def __init__(self, radius_bore: float, radius_port: float, angle: float = 0, pt1_time_constant_opening: float = 0.1, pt1_time_constant_closing: float = None, Cd: float = 0.64, **kwargs):
        super().__init__(**kwargs)
        self.FLE = FluidElement()
        self.radius_bore = radius_bore
        self.radius_port = radius_port
        self.__angle = angle
        self.angle_command = angle
        self.pt1_time_constant_opening = pt1_time_constant_opening
        self.pt1_time_constant_closing = pt1_time_constant_closing if pt1_time_constant_closing is not None else pt1_time_constant_opening
        self.angle_max = 90
        self.angle_min = 0
        self.Cd = Cd

    def angle_dot(self) -> float:
        angle_command_deg = self.angle_command * self.angle_max
        if angle_command_deg >= self.angle:
            angle_dot = (angle_command_deg - self.angle) / self.pt1_time_constant_opening
        else:
            angle_dot = (angle_command_deg - self.angle) / self.pt1_time_constant_closing
        return angle_dot

    @property
    def angle(self):
        return np.clip(self.__angle, self.angle_min, self.angle_max)

    @angle.setter
    def angle(self, value):
        self.__angle = value

    # def m_dot2(self):
    #     pressure_difference = self.delta_pressure() #- self.FLE.resistance(friction_coefficient, velocity, density, length, hydraulic_diameter)
    #     return pressure_difference / self.FLE.inductance(density=self.density(), length=1e-3, area=self.radius_port**2 * np.pi)
    # - implement valve resistance to use mass flow differential equation

    @property
    def m_dot(self):
        # valve area equation -> source: https://de.mathworks.com/help/hydro/ref/ballvalveil.html?searchHighlight=E&s_tid=doc_srchtitle
        lambda_bore = (self.radius_bore ** 2 - self.radius_port ** 2) / (2 * self.radius_bore)
        lambda_port = (self.radius_port ** 2 - self.radius_bore ** 2) / (2 * self.radius_port)
        port = self.radius_port ** 2 * (np.arccos(lambda_port) - lambda_port * np.sqrt(1 - lambda_port))
        bore = self.radius_bore ** 2 * (np.arccos(lambda_bore) - lambda_bore * np.sqrt(1 - lambda_bore) + port)
        area_valve = np.sin(np.deg2rad(self.angle)) * bore
        if area_valve < 1e-12:
            self.__m_dot = 0
            return self.__m_dot

        # --- Gas Flow Logic ---
        if 'gas' in self.phase() or 'supercritical' in self.phase():
            gamma = self.heat_capacity_ratio()
            p_ratio = self.node[-1].pressure / self.node[0].pressure
            p_ratio_choke = (2 / (gamma + 1)) ** (gamma / (gamma - 1))

            if p_ratio < p_ratio_choke:  # Choked gas flow
                self.__m_dot = self.Cd * area_valve * np.sqrt(gamma * self.density() * self.node[0].pressure * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)))
            else:  # Unchoked gas flow (subsonic)
                self.__m_dot = self.Cd * area_valve * np.sqrt(2 * self.density() * self.delta_pressure()) # Using incompressible form as an approximation for simplicity
        # --- Liquid Flow Logic ---
        elif 'liquid' in self.phase():
            try:
                # Get vapor pressure to check for cavitation
                p_vapor = CP.PropsSI('P', 'T', self.node[0].temperature(), 'Q', 0, self.system.fluid)
            except ValueError:
                p_vapor = 0 # No cavitation if vapor pressure is not defined (e.g., above critical point)
            
            if self.node[-1].pressure <= p_vapor: # Choked liquid flow (cavitation)
                dp_choked = self.node[0].pressure - p_vapor
                self.__m_dot = self.Cd * area_valve * np.sqrt(2 * self.density() * max(0, dp_choked))
            else: # Unchoked liquid flow
                self.__m_dot = self.Cd * area_valve * np.sqrt(2 * self.density() * self.delta_pressure())
        # --- Fallback/Two-Phase ---
        else:
             # Default to incompressible formula for two-phase or other states as a simplification
            self.__m_dot = self.Cd * area_valve * np.sqrt(2 * self.density() * self.delta_pressure())
        return self.__m_dot

    def density(self):
        return self.node[0].density()

    def phase(self):
        return self.node[0].phase()

    def heat_capacity_ratio(self):
        return self.node[0].heat_capacity_ratio()


topology = [
    Cavity(name='cavity_upstream', volume=100),
    Pipe(name='pipe_upstream', radius=0.025, length=1),
    Valve(name='valve', radius_bore=0.005, radius_port=0.005, pt1_time_constant_opening=0.3, Cd=0.9),
    Pipe(name='pipe_downstream', radius=0.025, length=1),
    Cavity(name='cavity_downstream', volume=0.001)
]



x = System(components = topology)

x.components[1].node[0] = x.components[0].node[-1]
x.components[2].node[0] = x.components[1].node[-1]
x.components[3].node[0] = x.components[2].node[-1]
x.components[4].node[0] = x.components[3].node[-1]

import scipy
result = scipy.integrate.solve_ivp(x.ode, t_span=(0, 0.5), y0=np.array([6e5, 0, 6e5, 0, 1e5, 0, 1e5]), method='RK45')

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

# class TestClass:
#     def __init__(self, **kwargs):
#         self.topo()
#         pass

#     def topo(self):
#             fluid = 'Propane'
#             self.system = FluidSystem(fluid=fluid)

    
#             self.cavity_upstream = Tank(system=self.system, volume=100)
#             self.pipe_upstream = Pipe(system=self.system, fluid=fluid, upstream=self.cavity_upstream, radius=0.025, length=1)
#             self.valve = Valve(system=self.system, fluid=fluid, upstream=self.pipe_upstream, radius_bore=0.025, radius_port=0.025, pt1_time_constant_opening=0.1, pt1_time_constant_closing=0.05, Cd=0.75)
#             self.pipe_downstream = Pipe(system=self.system, fluid=fluid, upstream=self.valve, radius=0.025, length=1)
#             self.cavity_downstream = Tank(system=self.system, fluid=fluid, upstream=self.pipe_downstream, volume=0.001)
#             print('Plant topology created')
#             pass

#     def ode(self, t, x):
#         ## Inputs
#         self.valve.angle_command = 0 if t < 0.1 else 1

#         ## States
#         self.cavity_upstream.pressure = x[0]
#         self.pipe_upstream.m_dot = x[1]
#         self.pipe_upstream.outlet.pressure = x[2]
#         self.valve.angle = x[3]
#         self.valve.outlet.pressure = x[4]
#         self.pipe_downstream.m_dot = x[5]
#         self.pipe_downstream.outlet.pressure = x[6]

#         ## DAE
#         self.valve.m_dot_update()

#         ## ODE
#         ode0 = self.cavity_upstream.p_dot()
#         ode1 = self.pipe_upstream.m_2dot()
#         ode2 = self.pipe_upstream.p_dot_out()
#         ode3 = self.valve.angle_dot()
#         ode4 = self.pipe_downstream.p_dot_in()
#         ode5 = self.pipe_downstream.m_2dot()
#         ode6 = self.cavity_downstream.p_dot()

#         return np.array([ode0, ode1, ode2, ode3, ode4, ode5, ode6])

# x = TestClass()

# import scipy
# result = scipy.integrate.solve_ivp(x.ode, t_span=(0, 0.5), y0=np.array([6e5, 0, 6e5, 0, 1e2, 0, 1e2]), method='RK45')

# import matplotlib.pyplot as plt
# plt.plot(result.t, result.y[0])
# # plt.plot(result.t, result.y[1])
# # plt.plot(result.t, result.y[2])
# plt.plot(result.t, result.y[3] / 90 * 6e5)
# # plt.plot(result.t, result.y[4])
# plt.plot(result.t, result.y[5])
# plt.plot(result.t, result.y[6])
# plt.show()



# ### TODO:
# - cavity = 1 node, capacitance
# - pipe = 1 link, inductance + resistance + capacitance
# - valve / constrict = 1 node, resistance
# - topo = nodes as inlet and outlets of components. capacitance universal for links and nodes. node universal for cap and res.
# - define cap, res and induct as own classes. Instantiate inside each component type
    
# - fix lump topology
#     - if node is outlet and component, volume is sum of both component volumes
# - implement flamettis line filling





    