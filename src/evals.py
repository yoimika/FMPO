from env import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

def tsne_2d(x):
    *prefix, N = x.shape
    flat = x.view(-1, N)

    flat_np = flat.detach().cpu().numpy()
    tsne = TSNE(n_components=2, init='random', random_state=0)
    flat_2d = tsne.fit_transform(flat_np)
    flat_2d = torch.from_numpy(flat_2d).to(x.device).float()
    ret_2d = flat_2d.view(*prefix, 2)
    return ret_2d

def handle_x_path(x_path):
    """Handle x_path for 2D action space."""
    T, N, act_dim = x_path.shape
    # act_dim > 2 -> 降维
    # act_dim < 2 -> 补零
    if act_dim > 2:
        x_path = tsne_2d(x_path)
    else:
        pad = np.zeros((T, N, 2 - act_dim))
        x_path = np.concatenate([x_path, pad], axis=-1)
    return x_path

def show_distribution(flow: Flow, env_config: EnvConfig):
    env_config = replace(env_config, vectorize_env=False)
    env_config = replace(env_config, render_mode=None)
    env = create_env(env_config, device=torch.device('cpu'))

    seed = 32
    env.reset(seed=seed)
    env.observation_space.seed(seed)

    obs, _ = env.reset()
    print(obs)
    env.close()
    N = 50
    obs = obs.repeat(N, 1)
    with torch.no_grad():
        obs = obs.to(flow.device)
        action, _, x_path = flow.sample_action(obs, True)
    action = action.cpu().numpy()
    x_path = x_path.cpu().numpy() # (steps, N, act_dim)

    # mu = torch.from_numpy(x_path[-1, ...]).float()
    # sigma = 0.3
    # x_path[-1, ...] = torch.normal(mu, sigma).numpy() 

    data = handle_x_path(x_path)

    # ---------- 用户可调 ----------
    single_w = 2.0          # 单幅宽度（英寸）
    single_h = 2.0          # 单幅高度（英寸）
    # ------------------------------

    T, N, _ = data.shape                       # data: (T, N, 2)
    colors = plt.cm.tab20(np.arange(N) % 20)   # 固定颜色

    fig, axes = plt.subplots(1, T,
                            figsize=(single_w * T, single_h),
                            sharey=True,
                            constrained_layout=False)  # 后面手动 tight

    if T == 1:
        axes = [axes]

    for t, ax in enumerate(axes):
        ax.scatter(data[t, :, 0], data[t, :, 1],
                c=colors, s=10, edgecolors='none')

        ax.set_xlim(-2.1, 2.1)      # 根据你的坐标范围改
        ax.set_ylim(-2.1, 2.1)
        ax.set_aspect('equal')
        ax.set_title(f't={t}')
        if t == 0:
            ax.set_ylabel('y')
        ax.set_xlabel('x')

    # 去掉子图间横向间距，纵向保留一点
    fig.subplots_adjust(wspace=0.05, hspace=0)
    plt.savefig('t_series_fixed_size.png', dpi=300)