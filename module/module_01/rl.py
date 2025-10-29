

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

class TrafficLightEnv:
    # 动作：0=STOP，1=GO（前进1格）
    def __init__(self, cfg=TLConfig()):
        self.cfg = cfg
        self.t = 0
        self.pos = 0
        self.done = False

    def reset(self):
        self.t = 0
        self.pos = 0
        self.done = False
        return self._obs()

    def _is_green(self):
        phase = self.t % (self.cfg.cycle_red + self.cfg.cycle_green)
        return phase >= self.cfg.cycle_red

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

    def _obs(self):
        return (self.pos, 1 if self._is_green() else 0)

    @property
    def n_states(self):
        return (self.cfg.road_len + 1) * 2

    @property
    def n_actions(self):
        return 2

    def s2i(self, s):
        pos, lg = s
        return pos * 2 + lg

# Q-learning快速训练
def train_q(env, episodes=320, alpha=0.5, gamma=0.95, eps_start=1.0, eps_end=0.05, eps_decay=0.985):
    Q = np.zeros((env.n_states, env.n_actions), dtype=np.float32)
    eps = eps_start
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

    def _select_action(self, s):
        if self.use_learned_policy:
            return int(np.argmax(self.Q[self.env.s2i(s)]))
        else:
            return np.random.randint(self.env.n_actions)

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

    def _step_once(self, force=False):
        if self.env.done:
            self.env.reset()
        s = self.env._obs()
        a = self._select_action(s) if (force or not self.paused) else 0  # 暂停时n键会force走一步
        self.env.step(a)
        self._render_frame()

    def anim_update(self, frame_idx):
        if not self.paused:
            self._step_once()
        return []

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

if __name__ == "__main__":
    main()



