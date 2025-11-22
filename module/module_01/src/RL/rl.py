#!/usr/bin/env python3
"""
一个极简的 Q-learning 示例：在一段直线路上驶向终点，途中遇到一个按红/绿循环变化的红绿灯。

核心概念
- 状态 state: (pos, light) = (车辆当前位置, 当前是否为绿灯: 0/1)
- 动作 action: 0=STOP（原地等待），1=GO（前进一格）
- 奖励 reward:
    - 靠近红灯却闯红灯（在灯前一格或灯位及之后且红灯时选择 GO）→ 大额惩罚 -10
    - 等待（STOP）→ 轻微惩罚 -0.1（鼓励尽快通过）
    - 抵达终点 → 奖励 +5
- 目标: 通过 Q-learning 学到一个策略，在尽可能少的时间内安全通过红绿灯到达终点。

可视化与交互
- 空格：暂停/继续
- n：单步推进一帧（暂停时也可强制走一步）
- g：在“学到的策略”和“随机策略”之间切换
- r：重置环境
- q：退出可视化窗口
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

# ---- 环境定义 ----
@dataclass
class TLConfig:
    road_len: int = 8          # 位置0..road_len，越过即到达
    light_pos: int = 5         # 红绿灯位置
    cycle_red: int = 20        # 红灯时长（tick）
    cycle_green: int = 20      # 绿灯时长（tick）
    max_steps: int = 60
    gamma: float = 0.95
    """交通灯环境的超参数配置。

    字段说明：
    - road_len: 道路长度；当 pos >= road_len 视为到达终点
    - light_pos: 红绿灯所在的位置坐标
    - cycle_red/cycle_green: 红/绿灯各自持续的 tick 数；二者周期相加构成完整循环
    - max_steps: 单回合的最大步数（防止无限循环）
    - gamma: 折扣因子（训练时也会传入 train_q）
    """

class TrafficLightEnv:
    # 动作：0=STOP，1=GO（前进1格）
    def __init__(self, cfg=TLConfig()):
        self.cfg = cfg
        self.t = 0
        self.pos = 0
        self.done = False
        """一个极简可离散化的交通灯环境。

        状态空间（离散）：
        - 位置 pos ∈ {0, 1, ..., road_len}
        - 灯色 light ∈ {0=Red, 1=Green}
        组合后总状态数为 (road_len+1) * 2。

        动作空间（离散）：
        - 0: STOP（等待）
        - 1: GO（向前移动 1 格）
        """

    def reset(self):
        self.t = 0
        self.pos = 0
        self.done = False
        return self._obs()
        """重置环境到初始状态并返回观测。"""

    def _is_green(self):
        phase = self.t % (self.cfg.cycle_red + self.cfg.cycle_green)
        return phase >= self.cfg.cycle_red
        """根据当前时间步 t 判断是否处于绿灯相位。"""

    def step(self, a):
        assert not self.done
        reward = 0.0

        if a == 1:  # GO
            # 接近灯且红灯则重罚；否则前进一步
            if (self.pos >= self.cfg.light_pos - 1) and (not self._is_green()):
                reward -= 10.0
            else:
                self.pos += 1
        else:       # STOP
            reward -= 0.1  # 等待的小惩罚

        if self.pos >= self.cfg.road_len:
            reward += 5.0
            self.done = True

        self.t += 1
        if self.t >= self.cfg.max_steps:
            self.done = True

        return self._obs(), reward, self.done, {}
        """执行一步交互。

        参数：
        - a: int, 动作（0=STOP, 1=GO）

        返回：
        - obs: tuple[int, int], (pos, is_green)
        - reward: float
        - done: bool, 是否回合结束
        - info: dict, 预留信息（此处为空字典）
        """

    def _obs(self):
        return (self.pos, 1 if self._is_green() else 0)
        """将环境内部状态转换为外部观测：位置与灯色（0/1）。"""

    @property #@property 装饰器将方法变为属性调用
    def n_states(self):
        return (self.cfg.road_len + 1) * 2
        """离散状态总数。"""

    @property #@property 装饰器将方法变为属性调用
    def n_actions(self):
        return 2
        """离散动作总数（STOP/GO）。"""

    def s2i(self, s):
        pos, lg = s
        return pos * 2 + lg
        """将 (pos, light) 的二元离散状态映射为一个索引，用于索引 Q 表。"""

# Q-learning快速训练
def train_q(env, episodes=320, alpha=0.5, gamma=0.95, eps_start=1.0, eps_end=0.05, eps_decay=0.985):
    Q = np.zeros((env.n_states, env.n_actions), dtype=np.float32)
    eps = eps_start
    """使用无模型 Q-learning 在离散状态-动作空间上学习 Q 表。

    参数：
    - env: TrafficLightEnv 实例
    - episodes: 训练回合数
    - alpha: 学习率（0~1）
    - gamma: 折扣因子（0~1）
    - eps_start/eps_end/eps_decay: ε-greedy 探索策略参数

    算法要点：
    - 动作选择：以概率 ε 随机探索，否则选择 Q 最大的贪心动作
    - Q 更新：Q[s,a] ← (1-α)Q[s,a] + α (r + γ max_a' Q[s',a'])
    - 每个 episode 结束后将 ε 按衰减率下降，最低不低于 eps_end
    """
    for _ in range(episodes):
        s = env.reset()
        while True:
            si = env.s2i(s)
            if np.random.rand() < eps:
                a = np.random.randint(env.n_actions)
            else:
                a = int(np.argmax(Q[si]))
            ns, r, done, _ = env.step(a)
            ni = env.s2i(ns)
            Q[si, a] = (1 - alpha) * Q[si, a] + alpha * (r + gamma * np.max(Q[ni]))
            s = ns
            if done:
                break
        eps = max(eps_end, eps * eps_decay)
    return Q

# 可视化绘制
class TLRenderer:
    def __init__(self, env, Q):
        self.env = env
        self.Q = Q
        self.use_learned_policy = True  # g键切换
        self.paused = False             # 空格切换
        self.fig, self.ax = plt.subplots(figsize=(7, 2.4))
        self._setup_canvas()
        self._connect_keys()
        """基于 matplotlib 的简单动画渲染器，用于演示学到的策略效果。"""

    def _setup_canvas(self):
        cfg = self.env.cfg
        self.ax.plot([0, cfg.road_len + 0.5], [0, 0], linewidth=6)   # 道路
        for p in range(cfg.road_len + 1):
            self.ax.plot([p, p], [-0.15, 0.15], linewidth=1)        # 刻度
        # 车（正方形点）
        self.car = self.ax.scatter([self.env.pos], [0], s=300, marker="s")
        self.car_label = self.ax.text(self.env.pos, -0.45, "Car", ha="center")
        # 红绿灯
        self.light = self.ax.scatter([cfg.light_pos], [0.5], s=400)
        self.light_txt = self.ax.text(cfg.light_pos, 0.85, "Light", ha="center", va="center")
        # 终点提示
        self.ax.text(cfg.road_len + 0.9, 0, "Goal →", va="center")
        # 顶部标题（动态更新）
        self.title = self.ax.set_title("Training loaded. Playing learned policy… (space: pause / n: step / g: toggle / r: reset / q: quit)")
        self.ax.set_xlim(-0.5, cfg.road_len + 1.8)
        self.ax.set_ylim(-1.0, 1.2)
        self.ax.axis("off")

    def _connect_keys(self):
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "n":
            self._step_once(force=True)
        elif event.key == "r":
            self.env.reset()
            self._render_frame()
        elif event.key == "g":
            self.use_learned_policy = not self.use_learned_policy
        elif event.key == "q":
            plt.close(self.fig)
        """处理键盘事件，实现简单的人机交互。"""

    def _select_action(self, s):
        if self.use_learned_policy:
            return int(np.argmax(self.Q[self.env.s2i(s)]))
        else:
            return np.random.randint(self.env.n_actions)
        """根据当前模式（学到的策略/随机）选择动作。"""

    def _render_frame(self):
        cfg = self.env.cfg
        # 更新灯颜色
        is_green = self.env._is_green()
        self.light.set_color("green" if is_green else "red")
        # 更新车位置
        self.car.set_offsets([[self.env.pos, 0]])
        self.car_label.set_position((self.env.pos, -0.45))
        # 更新标题
        self.title.set_text(
            f"{'LEARNED' if self.use_learned_policy else 'RANDOM'} | "
            f"t={self.env.t} | pos={self.env.pos} | light={'Green' if is_green else 'Red'} "
            f"(space pause / n step / g toggle / r reset / q quit)"
        )
        """重绘当前帧的可视化元素。"""

    def _step_once(self, force=False):
        if self.env.done:
            self.env.reset()
        s = self.env._obs()
        a = self._select_action(s) if (force or not self.paused) else 0  # 暂停时n键会force走一步
        self.env.step(a)
        self._render_frame()
        """推进环境一步，并刷新画面。

        参数：
        - force: True 时无视 paused 状态强制推进一步（用于单步按键）
        """

    def anim_update(self, frame_idx):
        if not self.paused:
            self._step_once()
        return []
        """matplotlib 动画回调：若未暂停则持续推进。"""

def main():
    cfg = TLConfig()
    env_for_train = TrafficLightEnv(cfg)
    Q = train_q(env_for_train, episodes=320, alpha=0.5, gamma=cfg.gamma)

    # 用一个新的env用于展示（避免训练尾状态影响）
    env = TrafficLightEnv(cfg)
    env.reset()

    renderer = TLRenderer(env, Q)
    # interval控制播放速度（毫秒）
    ani = FuncAnimation(renderer.fig, renderer.anim_update, interval=150, blit=False, cache_frame_data=False)
    plt.show()
    """训练一个 Q 表后，创建新的环境并以动画形式展示策略执行过程。"""

if __name__ == "__main__":
    main()


    
