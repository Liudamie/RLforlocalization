import numpy as np
import matplotlib.pyplot as plt


class ExtendedKalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = np.array(initial_state, dtype=float)  # 状态向量 [x, y]
        self.covariance = np.array(initial_covariance, dtype=float)  # 协方差矩阵
        self.process_noise = np.array(process_noise, dtype=float)  # 过程噪声矩阵
        self.measurement_noise = measurement_noise  # 测量噪声
        self.measurement_update = 0

    def predict(self, velocity, dt):
        F = np.array([[1, 0], [0, 1]])  # 状态转移矩阵
        B = np.array([[dt, 0], [0, dt]])  # 控制输入矩阵
        self.state = F @ self.state + B @ velocity
        self.covariance = F @ self.covariance @ F.T + self.process_noise

    def update(self, measurement):
        # 计算预期的相对距离
        distance = np.linalg.norm(self.state)
        # 防止除以零的情况，如果距离为0，则将雅可比矩阵设置为全零
        if distance != 0:

            H = (self.state / distance).reshape(1, 2)  # 计算雅可比矩阵，确保形状为 (1, 2)
        else:
            H = np.zeros((1, 2))  # 当距离为零时，雅可比矩阵无法定义，使用全零矩阵

        # 测量预测与测量噪声
        predicted_measurement = distance
        S = H @ self.covariance @ H.T + self.measurement_noise
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.measurement_update = measurement - predicted_measurement

        self.state = self.state +(K @ self.measurement_update.reshape(1, 1)).flatten()

        self.covariance = (np.eye(2) - K @ H) @ self.covariance


