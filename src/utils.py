import pinocchio as pin

def eepos_fk(model, data, q):
    pin.forwardKinematics(model, data, q)
    return data.oMi[6].translation

def eepos_fk_jac(model, data, q):
    pin.computeJointJacobians(model, data, q)
    
def euler(model, data, q, v, u, dt): 
    a = pin.aba(model, data, q, v, u)
    v_next = v + a * dt
    q_next = pin.integrate(model, q, v * dt)
    return q_next, v_next

def rk4(model, data, q, v, u, dt):    #model: pin.Model, data is pin.Data
    k1q = v
    k1v = pin.aba(model, data, q, v, u)
    q2 = pin.integrate(model, q, k1q * dt / 2)
    k2q = v + k1v * dt/2
    k2v = pin.aba(model, data, q2, k2q, u)
    q3 = pin.integrate(model, q, k2q * dt / 2)
    k3q = v + k2v * dt/2
    k3v = pin.aba(model, data, q3, k3q, u)
    q4 = pin.integrate(model, q, k3q * dt)
    k4q = v + k3v * dt
    k4v = pin.aba(model, data, q4, k4q, u)
    v_next = v + (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
    avg_v = (k1q + 2*k2q + 2*k3q + k4q) / 6
    q_next = pin.integrate(model, q, avg_v * dt)
    return q_next, v_next

def load_robot_model(urdf_path):
    model = pin.buildModelFromUrdf(urdf_path)
    visual_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL)
    collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION)
    
    return model, visual_model, collision_model

def print_stats(stats):
    for task, stat in stats.items():
        stat_list = stat['values']
        stat_unit = stat['unit']
        stat_mult = stat['multiplier'] # for conversions
        
        if not stat_list: # Skip empty
            continue
            
        avg_stat = stat_mult * sum(stat_list) / len(stat_list)
        min_stat, max_stat = stat_mult * min(stat_list), stat_mult * max(stat_list)
        
        print(f"{task}:")
        print(f"  avg: {avg_stat:.2f} {stat_unit}")
        print(f"  min: {min_stat:.2f} {stat_unit}") 
        print(f"  max: {max_stat:.2f} {stat_unit}")
        print()