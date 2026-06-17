import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
class NIPTMonteCarloSimulator:
    def __init__(self, optimal_times):
        """
        初始化模拟器
        Parameters:
        optimal_times: dict, 每个分组的最佳检测时间
        """
        self.optimal_times = optimal_times

    def delay_risk(self, t):
        """延迟风险函数"""
        return max(0, (20**(t-1)-1)/(20**14-1))

    def failure_risk(self, t):
        """失败风险函数 (假设为正态分布的概率)"""
        mean_failure_time = 13  # 假设失败风险的峰值在13周
        std_dev = 1.5  # 假设标准差为1.5
        z_score = (t - mean_failure_time) / std_dev
        return 1 - stats.norm.cdf(z_score)

    def total_risk(self, t, lambda_weight=0.8):
        """总风险函数"""
        delay_r = self.delay_risk(t)
        failure_r = self.failure_risk(t)
        return lambda_weight * delay_r + (1 - lambda_weight) * failure_r

    def monte_carlo_simulation(self, group, num_simulations=10000):
        """
        对某个分组进行蒙特卡洛模拟
        Parameters:
        group: str, 分组名称
        num_simulations: int, 模拟次数
        """
        optimal_time = self.optimal_times[group]
        simulated_times = np.random.normal(loc=optimal_time, scale=0.5, size=num_simulations)  # 假设检测时间服从正态分布
        simulated_times = np.clip(simulated_times, 10, 25)  # 限制检测时间在10到25周之间

        total_risks = [self.total_risk(t) for t in simulated_times]
        mean_risk = np.mean(total_risks)
        std_risk = np.std(total_risks)

        print(f"分组{group}:")
        print(f"  最优检测时间: {optimal_time:.4f}周")
        print(f"  模拟总风险均值: {mean_risk:.4f}")
        print(f"  模拟总风险标准差: {std_risk:.4f}")

        # 绘制风险分布图
        plt.figure(figsize=(8, 6))
        plt.hist(total_risks, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f"分组{group} - 总风险分布 (蒙特卡洛模拟)")
        plt.xlabel("总风险值")
        plt.ylabel("频数")
        plt.grid(alpha=0.3)
        plt.show()

    def run_simulation(self, num_simulations=10000):
        """
        对所有分组进行蒙特卡洛模拟
        Parameters:
        num_simulations: int, 模拟次数
        """
        for group in self.optimal_times.keys():
            self.monte_carlo_simulation(group, num_simulations=num_simulations)


def main():
    # 每个分组的最佳检测时间
    optimal_times = {
        "分组2": 12.2894,
        "分组0": 12.4423,
        "分组1": 12.5617,
        "分组3": 13.0738
    }

    # 初始化模拟器
    simulator = NIPTMonteCarloSimulator(optimal_times)

    # 运行蒙特卡洛模拟
    simulator.run_simulation(num_simulations=10000)


if __name__ == "__main__":
    main()