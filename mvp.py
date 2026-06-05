import numpy as np
import CoolProp.CoolProp as CP
import fluids
import matplotlib.pyplot as plt
import scipy

fluid = 'Propane'
temperature = 100
p_up = 6e5
p_down = 1e2
density = CP.PropsSI('D', 'T', temperature, 'P', p_up, fluid)
speed_of_sound = CP.PropsSI('A', 'T', temperature, 'P', p_up, fluid)
viscosity_dyn = CP.PropsSI('V', 'T', temperature, 'P', p_up, fluid)


v1 = 100
v2 = 1e-3

radius = 0.02
length = 0.5
area = radius**2 * np.pi
hydraulic_diameter = 4 * area / 2 * radius * np.pi
roughness = 1e-6
roughness_ratio = roughness / hydraulic_diameter

pt1_time_constant = 0.1
Cd = 0.9
radius_port = 0.001
radius_bore = radius_port

def ode(t, x):
    def inductance(density, length, area):
        return density * length / area

    def resistance(velocity, density, length, hydraulic_diameter):
        signed_square_velocity = velocity * np.abs(velocity)
        reynolds = density * (velocity + 1e-9) * hydraulic_diameter / viscosity_dyn
        friction_coefficient = fluids.friction.friction_factor(reynolds, eD=roughness_ratio, Method='Brkic_2011_2', Darcy=True)
        return friction_coefficient * length * density * signed_square_velocity / (2 * hydraulic_diameter)

    def capacitance(speed_of_sound, density, volume):
        bulk_modulus = speed_of_sound ** 2 * density
        return volume / bulk_modulus

    def m_dot2(delta_p, velocity, density, length, hydraulic_diameter, area):
        pressure_difference = delta_p - resistance(velocity, density, length, hydraulic_diameter)
        return pressure_difference / inductance(density, length, area)

    def p_dot(delta_m_dot, density, speed_of_sound, volume):
        return delta_m_dot / density / capacitance(speed_of_sound, density, volume)

    def v_a_dot(v_a_cmd, v_a, pt1_time_constant):
        angle_max = 90
        angle_command_deg = (v_a_cmd * angle_max)
        if angle_command_deg >= v_a:
            angle_dot = (angle_command_deg - v_a) / pt1_time_constant
        else:
            angle_dot = (angle_command_deg - v_a) / pt1_time_constant
        return angle_dot

    ## Inputs
    v_a_cmd = 0 if t < 0.1 else 1 # valve angle command

    ## States
    p1 = x[0]
    m1 = x[1]
    p2= x[2]
    v_a = x[3]
    p3 = x[4]
    m3 = x[5]
    p4 = x[6]

    ## DAE
    lambda_bore = (radius_bore ** 2 - radius_port ** 2) / (2 * radius_bore)
    lambda_port = (radius_port ** 2 - radius_bore ** 2) / (2 * radius_port)
    area_valve = np.sin(np.deg2rad(v_a)) * radius_bore ** 2 * (np.arccos(lambda_bore) - lambda_bore * np.sqrt(1 - lambda_bore) + radius_port ** 2 * (np.arccos(lambda_port) - lambda_port * np.sqrt(1 - lambda_port)))
    #area_valve = v_a * radius_port**2 * np.pi
    if p3 > p2:
        m2 = -Cd * area_valve * np.sqrt(2 * density * np.abs(p2-p3))
    else:
        m2 = Cd * area_valve * np.sqrt(2 * density * np.abs(p2-p3))

    ## ODE
    ode0 = p_dot(delta_m_dot=-m1, density=density, speed_of_sound=speed_of_sound, volume=v1)
    ode1 = m_dot2(delta_p=p1-p2, velocity=m2/density/area, density=density, length=length, hydraulic_diameter=hydraulic_diameter, area=area)
    ode2 = p_dot(delta_m_dot=m1-m2, density=density, speed_of_sound=speed_of_sound, volume=radius**2 * np.pi * length)
    ode3 = v_a_dot(v_a_cmd, v_a, pt1_time_constant)
    ode4 = p_dot(delta_m_dot=m2-m3, density=density, speed_of_sound=speed_of_sound, volume=radius**2 * np.pi * length)
    ode5 = m_dot2(delta_p=p3-p4, velocity=m2/density/area, density=density, length=length, hydraulic_diameter=hydraulic_diameter, area=area)
    ode6 = p_dot(delta_m_dot=m3, density=density, speed_of_sound=speed_of_sound, volume=v2)
    return np.array([ode0, ode1, ode2, ode3, ode4, ode5, ode6])

result = scipy.integrate.solve_ivp(ode, t_span=(0, 0.5), y0=np.array([p_up, 0, p_up, 0, p_down, 0, p_down]), method='RK45')

plt.plot(result.t, result.y[0] / 1e5, label='P1')
plt.plot(result.t, result.y[1], linestyle='--', label='m1')
plt.plot(result.t, result.y[2] / 1e5, label='P2')
#plt.plot(result.t, result.y[3] / 90 * 6, label='v')
plt.plot(result.t, result.y[4] / 1e5, label='P3')
plt.plot(result.t, result.y[5], linestyle='--', label='m2')
plt.plot(result.t, result.y[6] / 1e5, label='P4')

lambda_bore = (radius_bore ** 2 - radius_port ** 2) / (2 * radius_bore)
lambda_port = (radius_port ** 2 - radius_bore ** 2) / (2 * radius_port)
area_valve = np.sin(np.deg2rad(result.y[3])) * radius_bore ** 2 * (np.arccos(lambda_bore) - lambda_bore * np.sqrt(1 - lambda_bore) + radius_port ** 2 * (np.arccos(lambda_port) - lambda_port * np.sqrt(1 - lambda_port)))
plt.plot(result.t, area_valve /(radius_port**2 * np.pi) * 6, label='area')

plt.legend()
plt.show()