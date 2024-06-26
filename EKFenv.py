import gym
from gym import spaces
import numpy as np
import ekfdata
class EKFTrackingEnv(gym.Env):
    def __init__(self):
        super(EKFTrackingEnv, self).__init__()
        self.action_space = spaces.Box(low=-0.01, high=0.01, shape=(2,1), dtype=np.float32)  # 控制输入：速度向量
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,1), dtype=np.float32)  # 状态空间

        # EKF 初始化
        initial_state = [0.0, 0.0]
        initial_covariance = [[1, 0], [0, 1]]
        process_noise = [[0.01, 0], [0, 0.01]]
        measurement_noise = 0.01
        self.ekf = ekfdata.ExtendedKalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise)
        self.dt = 0.1  # 时间步长

        # 真实位置初始化
        self.true_state = np.array([0.0, 0.0])

    def reset(self):
        self.ekf.state = np.array([0.0, 0.0])
        self.ekf.covariance = np.array([[1, 0], [0, 1]])
        self.true_state = np.array([0.0, 0.0])
        return self.ekf.state

    def step(self, action):
        # 真实位置更新
        #self.true_state += action * self.dt
        velocity = np.random.normal(0, 1, 2)
       # action = np.clip(action, -0.1, 0.1)

        angle = np.random.uniform(0, 2 * np.pi)

        # 根据角度计算速度向量的分量
        vx = 1 * np.cos(angle)
        vy = 1 * np.sin(angle)
        # 形成速度向量
        velocity = np.array([vx, vy])

        # EKF 预测
        estimate_velocity = velocity +np.random.normal(0,0.03,2)
        self.ekf.predict(velocity=estimate_velocity, dt=self.dt)
        self.true_state = self.true_state + velocity * self.dt
        # 模拟观测：真实距离加上测量噪声
        true_distance = np.linalg.norm(self.true_state)
        measurement = true_distance + np.random.normal(0, np.sqrt(self.ekf.measurement_noise))

        # EKF 更新
        self.ekf.update(measurement=measurement)


        # 计算奖励：真实位置与估计位置的误差的负值

        estimated_position = self.ekf.state + (np.transpose(action) @ self.ekf.measurement_update.reshape(1, 1)).flatten()
        self.ekf.state = estimated_position
        reward = -np.linalg.norm(self.true_state - estimated_position)

        self.ekf.state = estimated_position
        # 这里简化结束条件，可以根据需要设置

        done = np.linalg.norm(self.true_state) > 3

        return self.ekf.state, reward, done, {}

    def render(self, mode='human'):
        # 可视化（可选实现）
        pass

    def close(self):
        pass
