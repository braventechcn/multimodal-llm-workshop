# encoding=utf-8
"""
在无显示环境（如远程 Linux 服务器）下保存 RL 动画为 GIF。
输出：module/module_01/outputs/traffic_light.gif
依赖：matplotlib[pillow]、numpy
"""

import os
from pathlib import Path

# 使用非交互后端，避免需要 GUI 显示
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

# 从同目录导入 rl 中的环境、训练与渲染类
from rl import TLConfig, TrafficLightEnv, train_q, TLRenderer


def main(out_path: str = "module/module_01/outputs/traffic_light.gif", frames: int = 200, fps: int = 8):
    cfg = TLConfig()
    env_for_train = TrafficLightEnv(cfg)
    Q = train_q(env_for_train, episodes=320, alpha=0.5, gamma=cfg.gamma)

    env = TrafficLightEnv(cfg)
    env.reset()

    renderer = TLRenderer(env, Q)
    ani = FuncAnimation(
        renderer.fig,
        renderer.anim_update,
        frames=frames,
        interval=150,
        blit=False,
        cache_frame_data=False,
        repeat=False,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out), writer="pillow", fps=fps)
    print(f"动画已保存为: {out} ({frames} 帧, {fps} FPS)")


if __name__ == "__main__":
    # 允许通过环境变量覆盖输出路径、帧数、FPS
    out = os.environ.get("RL_OUT", "module/module_01/outputs/traffic_light_demo.gif")
    frames = int(os.environ.get("RL_FRAMES", "200"))
    fps = int(os.environ.get("RL_FPS", "8"))
    main(out, frames, fps)
