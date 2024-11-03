import numpy as np
from numba import jit

np.random.seed(3407)

DECIMAL_TYPE = np.float64
NOPYTHON = True
FASTMATH = True

# 配置参数
G = 6.67430e-11  # 万有引力常数
c = 299792458  # 光速，单位：米/秒
dt = 3600  # 时间步长，单位：秒
total_time = 30 * 24 * 3600  # 总模拟时间，单位：秒


num_planets = 4  # 假设我们有4个行星
softening = 1e3  # 软ening 参数，防止除以零

# 使用双精度浮点数
planet_masses = np.random.uniform(1e20, 1e30, num_planets).astype(DECIMAL_TYPE)
planet_positions = np.random.uniform(-1e12, 1e12, (num_planets, 3)).astype(DECIMAL_TYPE)
planet_velocities = np.random.uniform(-1e4, 1e4, (num_planets, 3)).astype(DECIMAL_TYPE)
planet_rotational_speeds = np.random.uniform(-1e-5, 1e-5, num_planets).astype(
    DECIMAL_TYPE
)


def add_planet():
    global planet_masses, planet_positions, planet_velocities, planet_rotational_speeds
    planet_masses = np.append(planet_masses, 7.348e22)  # 添加的行星质量
    planet_positions = np.append(
        planet_positions, [[20e11, 0, 0]], axis=0
    )  # 添加的行星位置
    planet_velocities = np.append(
        planet_velocities, [[0, 10e3, 0]], axis=0
    )  # 添加的行星速度
    planet_rotational_speeds = np.append(
        planet_rotational_speeds, 1e-5
    )  # 添加的行星自转速度


# 计算距离
@jit(nopython=NOPYTHON, fastmath=FASTMATH)
def distance(pos1, pos2):
    return np.linalg.norm(pos2 - pos1)


@jit(nopython=NOPYTHON, fastmath=FASTMATH)
def acceleration(positions, masses, rotational_speeds):
    num_planets = len(masses)
    accelerations = np.zeros((num_planets, 3))
    for i in range(num_planets):
        for j in range(num_planets):
            if i != j:
                r = distance(positions[i], positions[j])
                if r == 0:
                    continue  # 避免除以零
                a = G * masses[j] / ((r**2) + softening)
                direction = (positions[j] - positions[i]) / r
                accelerations[i] += a * direction

                # 添加潮汐力
                tidal_force = (
                    G * masses[j] * (positions[i] - positions[j]) / ((r**3) + softening)
                )
                accelerations[i] += tidal_force

                # 添加自转效应
                rotational_force = np.cross(
                    rotational_speeds[i] * positions[i], direction
                )
                accelerations[i] += rotational_force

                # 添加相对论效应
                relativity_factor = 1 + (G * masses[j] / (c**2 * r))
                accelerations[i] *= relativity_factor
    return accelerations


# 更新位置和速度
@jit(nopython=NOPYTHON, fastmath=FASTMATH)
def update_positions(positions, velocities, accelerations, dt):
    velocities += accelerations * dt
    positions += velocities * dt
    return positions, velocities


# 检查碰撞
@jit(nopython=NOPYTHON, fastmath=FASTMATH)
def check_collision(positions, velocities, masses, threshold=1e9):
    num_planets = len(masses)
    for i in range(num_planets):
        for j in range(i + 1, num_planets):
            if distance(positions[i], positions[j]) < threshold:
                # 计算碰撞后的速度
                v_i = velocities[i]
                v_j = velocities[j]
                m_i = masses[i]
                m_j = masses[j]
                velocities[i] = ((m_i - m_j) * v_i + 2 * m_j * v_j) / (m_i + m_j)
                velocities[j] = ((m_j - m_i) * v_j + 2 * m_i * v_i) / (m_i + m_j)

                # 计算碰撞后的质量
                masses[i] = m_i + m_j
                masses[j] = masses[i]

                # 将发生碰撞的行星位置移动开，防止再次碰撞
                positions[i] += (
                    threshold
                    * (positions[i] - positions[j])
                    / distance(positions[i], positions[j])
                )
                positions[j] += (
                    threshold
                    * (positions[j] - positions[i])
                    / distance(positions[i], positions[j])
                )
    return positions, velocities, masses


def check_regularity(history, planet_index, tolerance=1e-6):
    positions = np.array(history["position"])
    planet_positions = positions[:, planet_index % 3]
    variance = np.var(planet_positions, axis=0)
    return variance < tolerance


# 检查引力捕获
@jit(nopython=NOPYTHON, fastmath=FASTMATH)
def check_gravitational_capture(positions, velocities, masses, threshold=1e10):
    num_planets = len(masses)
    for i in range(num_planets):
        for j in range(i + 1, num_planets):
            r = distance(positions[i], positions[j])
            v = distance(velocities[i], velocities[j])
            if r < threshold and v < threshold:
                # 引力捕获
                velocities[i] = (
                    masses[j] * velocities[j] + masses[i] * velocities[i]
                ) / (masses[i] + masses[j])
                velocities[j] = velocities[i]
    return velocities


# 检查潮汐锁定
@jit(nopython=NOPYTHON, fastmath=FASTMATH)
def check_tidal_locking(positions, velocities, rotational_speeds, dt):
    num_planets = len(rotational_speeds)
    for i in range(num_planets):
        for j in range(num_planets):
            if i != j:
                r = distance(positions[i], positions[j])
                if r == 0:
                    continue  # 避免除以零
                tidal_force_magnitude = G * planet_masses[j] / ((r**3) + softening)
                tidal_torque = np.cross(
                    positions[i] - positions[j],
                    tidal_force_magnitude * (positions[i] - positions[j]),
                )
                rotational_speeds[i] -= (
                    np.linalg.norm(tidal_torque) * dt / planet_masses[i]
                )
    return rotational_speeds


# 检查轨道共振
def check_orbital_resonance(velocities, tolerance=1e-2):
    num_planets = len(velocities)
    for i in range(num_planets):
        for j in range(i + 1, num_planets):
            orbital_ratio = velocities[i][1] / velocities[j][1]
            if abs(orbital_ratio - round(orbital_ratio)) < tolerance:
                return True
    return False


def simlator():
    global planet_positions, planet_velocities, planet_masses, planet_rotational_speeds
    for t in range(int(total_time / dt)):
        accelerations = acceleration(
            planet_positions, planet_masses, planet_rotational_speeds
        )
        planet_positions, planet_velocities = update_positions(
            planet_positions, planet_velocities, accelerations, dt
        )
        planet_positions, planet_velocities, planet_masses = check_collision(
            planet_positions, planet_velocities, planet_masses
        )
        planet_velocities = check_gravitational_capture(
            planet_positions, planet_velocities, planet_masses
        )
        planet_rotational_speeds = check_tidal_locking(
            planet_positions, planet_velocities, planet_rotational_speeds, dt
        )

        for i in range(len(planet_masses)):
            yield {"planet": i, "time": t, "pos": planet_positions[i]}

        # 将计算出的加速度应用到原始加速度上
        planet_positions, planet_velocities = update_positions(
            planet_positions, planet_velocities, accelerations, dt
        )


if __name__ == "__main__":
    history = {"planet": [], "time": [], "position": []}
    for data in simlator():
        print(f"Planet {data['planet']}(time: {data['time']}) position: {data['pos']}")
        history["planet"].append(data["planet"])
        history["time"].append(data["time"])
        history["position"].append(data["pos"])

    # 检查规律性
    planet_indices = range(len(planet_masses))  # 所有行星的索引
    regularities = [
        check_regularity(history, planet_index) for planet_index in planet_indices
    ]
    print(f"All planets have regular positions: {all(regularities)}")

    # 检查轨道共振
    orbital_resonance = check_orbital_resonance(planet_velocities)
    print(f"Orbital resonance detected: {orbital_resonance}")
