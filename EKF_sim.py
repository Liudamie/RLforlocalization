import numpy as np
import matplotlib.pyplot as plt
import ekfdata
initial_state = [1, 1]
initial_covariance = [[1, 0], [0, 1]]
process_noise = [[0.1, 0], [0, 0.1]]
measurement_noise = 0.01

ekf = ekfdata.ExtendedKalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise)

# 模拟数据
time_steps = 1000
dt = 0.1
velocity_noise_std = 1
measurement_noise_std = 0.2

true_positions = [np.array(initial_state)]
estimated_positions = [np.array(initial_state)]

for _ in range(time_steps):
    # 随机生成相对速度
    velocity = np.random.normal(0, velocity_noise_std, 2)

    # 更新真实位置
    true_position = true_positions[-1] + velocity * dt
    true_positions.append(true_position)

    # 获取带噪声的测量：相对距离
    measurement = np.linalg.norm(true_position) + np.random.normal(0, measurement_noise_std)

    # EKF预测和更新
    ekf.predict(velocity +np.random.normal(0,0.03,2), dt)
    ekf.update(measurement)
    estimated_positions.append(ekf.state.copy())

# 绘图比较
true_positions = np.array(true_positions)
estimated_positions = np.array(estimated_positions)

# 计算RMSE
errors = true_positions - estimated_positions
mse = np.mean(errors**2, axis=0)
rmse = np.sqrt(mse)
print(f"RMSE for X Position: {rmse[0]}")
print(f"RMSE for Y Position: {rmse[1]}")
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(true_positions[:, 0], 'g-', label='True X Position')
plt.plot(estimated_positions[:, 0], 'b--', label='Estimated X Position')
plt.xlabel('Time Step')
plt.ylabel('X Position')
plt.title('True vs Estimated X Position')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(true_positions[:, 1], 'g-', label='True Y Position')
plt.plot(estimated_positions[:, 1], 'b--', label='Estimated Y Position')
plt.xlabel('Time Step')
plt.ylabel('Y Position')
plt.title('True vs Estimated Y Position')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

