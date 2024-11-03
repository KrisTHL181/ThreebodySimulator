import numpy as np
from numba import jit

np.random.seed(3407)

DECIMAL_TYPE = np.float64
NOPYTHON = True
FASTMATH = True

# 配置参数
G = 6.67430e-11  # 万有引力常数
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
    return accelerations


# 更新位置和速度
@jit(nopython=NOPYTHON, fastmath=FASTMATH)
def update_positions(positions, velocities, accelerations, dt):
    velocities += accelerations * dt
    positions += velocities * dt
    return positions, velocities


def check_regularity(history, planet_index, tolerance=1e-6):
    positions = np.array(history["position"])
    planet_positions = positions[:, planet_index % 3]
    variance = np.var(planet_positions, axis=0)
    return variance < tolerance


def simlator():
    global planet_positions, planet_velocities
    for t in range(int(total_time / dt)):
        accelerations = acceleration(
            planet_positions, planet_masses, planet_rotational_speeds
        )
        planet_positions, planet_velocities = update_positions(
            planet_positions, planet_velocities, accelerations, dt
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
