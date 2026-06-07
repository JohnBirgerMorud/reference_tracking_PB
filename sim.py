import os
import sys

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
from matplotlib.widgets import Slider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, BASE_DIR)

from experiment_params import getCarFinalParams, getCarInitParams, getLossParams
from config import device
from controllers import PerfBoostController
from controllers.MLP import ZeroController
from loss_functions import BumpercarLoss
from plants import BumpercarSystem, car_params
from plants.bumpercar.bumpercar_dataset import BumpercarDataset


# Add your trained pRB checkpoint here. Prefer a checkpoint with MLP weights, e.g.
# "experiments/robots/saved_results/perf_boost_XX_XX_XX_XX_XX/checkpoints/checkpoint_latest.pt"
# TRAINED_PBR_MODEL_PATH = "experiments/bumpercar/trained_pRB/controller_smoother_traj.pt"
# TRAINED_PBR_MODEL_PATH = "experiments/bumpercar/trained_pRB/controller_zero_collisions.pt"
# TRAINED_PBR_MODEL_PATH = "experiments/bumpercar/trained_pRB/checkpoint_epoch_00110 (2) copy.pt"
# TRAINED_PBR_MODEL_PATH = "experiments/bumpercar/trained_pRB/checkpoint_epoch_00110 (2) copy.pt"
TRAINED_PBR_MODEL_PATH = "experiments/bumpercar/trained_pRB/final_controllers_1.pt"
# TRAINED_PBR_MODEL_PATH = "experiments/bumpercar/trained_pRB/trained_controller_loss227_trueArena.pt"

EVALUATE_MODEL = True
EVAL_HORIZON = 500
EVAL_NUM_ROLLOUTS = 100
EVAL_NUM_TEST_ROLLOUTS = 500
EVAL_RANDOM_SEED = 2

SIM_USE_GENERATED_SAMPLE = True
SIM_DATA_SPLIT = "train"
SIM_SAMPLE_INDEX = 32
SIM_RANDOM_SEED = EVAL_RANDOM_SEED
OBSTACLE_RADIUS = 1.5
REPORT_FIGURE_PATH = "experiments/bumpercar/report_trajectory.svg"
REPORT_FINAL_RADIUS = 1.0

DT = 0.04


def make_generated_sample_data(horizon, sample_index=0, random_seed=11, split="train"):
    train_data, test_data = make_eval_data(
        horizon=horizon,
        num_rollouts=max(sample_index + 1, 1),
        num_test_rollouts=max(sample_index + 1, 1),
        random_seed=random_seed,
    )

    if split == "train":
        source_data = train_data
    elif split == "test":
        source_data = test_data
    else:
        raise ValueError(f"Unknown generated sample split: {split}")

    data = source_data[sample_index:sample_index + 1]
    nx = data.shape[-1] // 2
    xbar = data[0, 0, nx:]

    return data.to(device), xbar.to(device)


def make_crossing_data(horizon, n_agents=2, ):
    nx = 7 * n_agents
    data = torch.zeros(1, horizon, 2 * nx)

    x0_bumpercar, x_final_bumpercar, _ , _, _, _ = getCarInitParams(device)
    
    data[:, 0, :nx] = x0_bumpercar
    data[:, :, nx:] = x_final_bumpercar.view(1, 1, -1)
    
    return data.to(device), x_final_bumpercar


def make_eval_data(horizon, num_rollouts, num_test_rollouts, random_seed):
    x0_bumpercar, x_final_bumpercar, _ , _, car_init_radius, std_init_theta = getCarInitParams(device)
    x_final_limit, y_final_limit, final_car_min_dist = getCarFinalParams()
    
    dataset = BumpercarDataset(
        random_seed=random_seed,
        horizon=horizon,
        x0=x0_bumpercar,
        x_final=x_final_bumpercar,
        car_init_radius=car_init_radius,
        x_final_limit=x_final_limit,
        y_final_limit=y_final_limit,
        final_car_min_dist=final_car_min_dist,
        std_init_theta = std_init_theta,
        n_agents=2,
    )
    train_data, test_data = dataset.get_data(
        num_train_samples=num_rollouts,
        num_test_samples=num_test_rollouts,
    )
    return train_data.to(device), test_data.to(device)


def make_loss_fn(train_data, n_agents=2):
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)    
    Q, Q_final, Qs, alpha_col, alpha_obst, alpha_u, position_deadzone, steady_state_velocity_radius, min_dist = getLossParams(device)
    
    return BumpercarLoss(
        Q=Q,
        Q_final=Q_final,
        Qs=Qs,
        alpha_u=alpha_u,
        xbar=train_data[0, :, 14:],
        loss_bound=None,
        sat_bound=None,
        alpha_col=alpha_col,
        alpha_obst=alpha_obst,
        obstacle_centers=obstacle_centers,
        obstacle_covs=obstacle_covs,
        min_dist=min_dist,
        n_agents=n_agents,
        position_deadzone=position_deadzone,
        steady_state_velocity_radius=steady_state_velocity_radius,
    )


def obstacle_collision_indices(x_log, obstacle_centers, radius=OBSTACLE_RADIUS, n_agents=2):
    collision_indices = set()
    radius_sq = radius ** 2

    for agent_idx in range(n_agents):
        base = 7 * agent_idx
        positions = x_log[:, :, base:base + 2]

        for obstacle_idx, center in enumerate(obstacle_centers):
            center = center.to(device=positions.device, dtype=positions.dtype).view(1, 1, 2)
            distance_sq = torch.sum((positions - center) ** 2, dim=-1)
            colliding = distance_sq <= radius_sq

            for rollout_idx, _ in colliding.nonzero(as_tuple=False):
                collision_indices.add(int(rollout_idx))

    return sorted(collision_indices)


def checkpoint_args(checkpoint):
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else vars(args)


def load_controller(system, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint_args(checkpoint)

    controller = PerfBoostController(
        noiseless_forward=system.noiseless_forward,
        input_init=system.x_init,
        output_init=system.u_init,
        dim_internal=args.get("dim_internal", 8),
        dim_nl=args.get("dim_nl", 8),
        initialization_std=args.get("cont_init_std", 0.1),
        output_amplification=1,
        ren_internal_state_init=None,
    ).to(device)

    if "controller_state_dict" in checkpoint:
        controller.load_state_dict(checkpoint["controller_state_dict"], strict=False)
    elif "ren_state_dict" in checkpoint:
        controller.c_ren.load_state_dict(checkpoint["ren_state_dict"], strict=False)
        if "mlp_state_dict" in checkpoint:
            controller.MLP.load_state_dict(checkpoint["mlp_state_dict"], strict=False)
    else:
        ren_state = {
            key: value for key, value in checkpoint.items()
            if key in controller.c_ren.state_dict()
        }
        controller.c_ren.load_state_dict(ren_state, strict=False)

    controller.eval()
    controller.reset()
    return controller


def evaluate_controller():
    if not TRAINED_PBR_MODEL_PATH:
        raise ValueError("Set TRAINED_PBR_MODEL_PATH before evaluating.")

    system = BumpercarSystem(
        params=car_params,
        x_init=None,
        u_init=None,
        dt=DT,
    ).to(device)
    controller = load_controller(system, TRAINED_PBR_MODEL_PATH)
    train_data, test_data = make_eval_data(
        horizon=EVAL_HORIZON,
        num_rollouts=EVAL_NUM_ROLLOUTS,
        num_test_rollouts=EVAL_NUM_TEST_ROLLOUTS,
        random_seed=EVAL_RANDOM_SEED
    )
    loss_fn = make_loss_fn(train_data, n_agents=system.n_agents)

    print(f"[INFO] evaluating {TRAINED_PBR_MODEL_PATH}")
    _, _, obstacle_centers, _, _, _ = getCarInitParams(device)
    with torch.no_grad():
        x_log, e_log, u_log = system.rollout(controller, train_data, train=False)
        train_loss = loss_fn.forward(x_log, u_log, e_log).item()
        train_collisions = loss_fn.count_collisions(x_log)
        train_obstacle_collision_indices = obstacle_collision_indices(
            x_log,
            obstacle_centers,
            n_agents=system.n_agents,
        )

        x_log, e_log, u_log = system.rollout(controller, test_data, train=False)
        test_loss = loss_fn.forward(x_log, u_log, e_log).item()
        test_collisions = loss_fn.count_collisions(x_log)
        test_obstacle_collision_indices = obstacle_collision_indices(
            x_log,
            obstacle_centers,
            n_agents=system.n_agents,
        )

    print(f"Train loss: {train_loss:.4f} -- Number of collisions = {train_collisions:.0f}")
    print(f"Test loss: {test_loss:.4f} -- Number of collisions = {test_collisions:.0f}")
    print(f"Train obstacle collision indices: {train_obstacle_collision_indices}")
    print(f"Test obstacle collision indices: {test_obstacle_collision_indices}")


def simulate(horizon=400, use_generated_sample=SIM_USE_GENERATED_SAMPLE, sample_index=SIM_SAMPLE_INDEX):
    system = BumpercarSystem(
        params=car_params,
        x_init=None,
        u_init=None,
        dt = DT,
    ).to(device)

    if TRAINED_PBR_MODEL_PATH:
        controller = load_controller(system, TRAINED_PBR_MODEL_PATH)
        title = "Trained pRB controller"
    else:
        controller = ZeroController(ref_dim=2 * system.n_agents).to(device)
        title = "PID only - set TRAINED_PBR_MODEL_PATH to use pRB"

    if use_generated_sample:
        data, xbar = make_generated_sample_data(
            horizon=horizon,
            sample_index=sample_index,
            random_seed=SIM_RANDOM_SEED,
            split=SIM_DATA_SPLIT,
        )
        title = f"{title} - {SIM_DATA_SPLIT} generated sample {sample_index}"
    else:
        data, xbar = make_crossing_data(horizon=horizon, n_agents=system.n_agents)
        title = f"{title} - fixed crossing"

    with torch.no_grad():
        x_log, _, _ = system.rollout(controller, data, train=False)

    _, _, obstacle_centers, _, _, _ = getCarInitParams(device)
    sim_obstacle_collision_indices = obstacle_collision_indices(
        x_log,
        obstacle_centers,
        n_agents=system.n_agents,
    )
    # print(f"Sim obstacle collision indices: {sim_obstacle_collision_indices}")

    return x_log[0].detach().cpu(), xbar.cpu(), title


def draw_car(ax, x, y, theta, color):
    length = 0.30
    width = 0.16
    dx = torch.tensor([length / 2, length / 2, -length / 2, -length / 2])
    dy = torch.tensor([width / 2, -width / 2, -width / 2, width / 2])
    c = torch.cos(theta)
    s = torch.sin(theta)
    px = x + c * dx - s * dy
    py = y + s * dx + c * dy
    return ax.fill(px, py, color=color, alpha=0.75, edgecolor="black", linewidth=1.0)[0]


def draw_pose_arrow(ax, x, y, theta, color, length=0.45):
    dx = length * torch.cos(theta).item()
    dy = length * torch.sin(theta).item()
    arrow = FancyArrowPatch(
        (x.item(), y.item()),
        (x.item() + dx, y.item() + dy),
        arrowstyle="-|>",
        mutation_scale=12,
        color=color,
        linewidth=1.2,
        zorder=5,
    )
    ax.add_patch(arrow)
    return arrow


def draw_obstacles(ax, obstacle_centers, obstacle_covs):
    for center, _ in zip(obstacle_centers, obstacle_covs):
        center = center.detach().cpu().flatten()

        circle = Circle(
            xy=(center[0].item(), center[1].item()),
            radius=OBSTACLE_RADIUS,
            facecolor="0.35",
            edgecolor="black",
            alpha=0.25,
            linewidth=1.0,
            zorder=0,
        )
        ax.add_patch(circle)


def draw_sample_regions(ax, colors):
    x0_bumpercar, x_final, _, _, car_init_radius, _ = getCarInitParams(device)

    for i, color in enumerate(colors):
        base = 7 * i
        init_circle = Circle(
            xy=(x0_bumpercar[base].item(), x0_bumpercar[base + 1].item()),
            radius=car_init_radius,
            facecolor=color,
            edgecolor=color,
            alpha=0.10,
            linewidth=1.2,
            zorder=0,
        )
        final_circle = Circle(
            xy=(x_final[base].item(), x_final[base + 1].item()),
            radius=REPORT_FINAL_RADIUS,
            facecolor=color,
            edgecolor=color,
            alpha=0.08,
            linestyle="--",
            linewidth=1.2,
            zorder=0,
        )
        ax.add_patch(init_circle)
        ax.add_patch(final_circle)


def save_report_figure(path=REPORT_FIGURE_PATH):
    x_log, xbar, _ = simulate()
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    n_agents = 2
    colors = ["tab:blue", "tab:orange"]

    fig, ax = plt.subplots(figsize=(6.2, 7.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-4.0, 6.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)

    draw_sample_regions(ax, colors)
    draw_obstacles(ax, obstacle_centers, obstacle_covs)

    for i in range(n_agents):
        base = 7 * i
        color = colors[i]
        ax.plot(x_log[:, base], x_log[:, base + 1], color=color, linewidth=2.0, linestyle="--")
        ax.plot(x_log[0, base], x_log[0, base + 1], marker="o", markersize=6, color=color, fillstyle="none")
        ax.plot(xbar[base], xbar[base + 1], marker="*", markersize=12, color=color)
        draw_car(ax, x_log[-1, base], x_log[-1, base + 1], x_log[-1, base + 2], color)
        draw_pose_arrow(ax, x_log[-1, base], x_log[-1, base + 1], x_log[-1, base + 2], color)

    legend_handles = [
        Line2D([0], [0], color="0.25", linewidth=2.0, linestyle="--", label="Trajectory"),
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", markersize=7, fillstyle="none", label="Initial position"),
        Line2D([0], [0], marker="*", color="0.25", linestyle="None", markersize=12, label="Final target"),
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", markersize=13, fillstyle="none", label="Sample region"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.95)

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved report figure to {path}")


def show_simulation():
    x_log, xbar, title = simulate()
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    _, _, _, _, _, _, _, _, min_dist = getLossParams(device)
    n_agents = 2
    colors = ["tab:blue", "tab:orange"]

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.16)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-4.0, 6.0)
    ax.grid(True, alpha=0.3)
    draw_obstacles(ax, obstacle_centers, obstacle_covs)

    path_lines = []
    car_patches = []
    safety_patches = []
    target_markers = []
    start_markers = []

    for i in range(n_agents):
        base = 7 * i
        color = colors[i]
        (path_line,) = ax.plot([], [], color=color, linewidth=2)
        path_lines.append(path_line)
        car_patches.append(draw_car(ax, x_log[0, base], x_log[0, base + 1], x_log[0, base + 2], color))
        safety_circle = Circle(
            xy=(x_log[0, base].item(), x_log[0, base + 1].item()),
            radius=min_dist / 2,
            facecolor=color,
            edgecolor=color,
            alpha=0.08,
            linewidth=1.0,
        )
        ax.add_patch(safety_circle)
        safety_patches.append(safety_circle)
        target_markers.append(ax.plot(xbar[base], xbar[base + 1], marker="*", markersize=14, color=color)[0])
        start_markers.append(ax.plot(x_log[0, base], x_log[0, base + 1], marker="o", markersize=7, color=color, fillstyle="none")[0])

    time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")

    slider_ax = fig.add_axes([0.15, 0.05, 0.72, 0.035])
    time_slider = Slider(
        ax=slider_ax,
        label="time",
        valmin=0,
        valmax=x_log.shape[0] - 1,
        valinit=0,
        valstep=1,
    )

    def update(frame):
        t = int(frame)
        for i in range(n_agents):
            base = 7 * i
            path_lines[i].set_data(x_log[:t + 1, base], x_log[:t + 1, base + 1])
            safety_patches[i].center = (x_log[t, base].item(), x_log[t, base + 1].item())
            car_patches[i].remove()
            car_patches[i] = draw_car(
                ax,
                x_log[t, base],
                x_log[t, base + 1],
                x_log[t, base + 2],
                colors[i],
            )

        dist = torch.linalg.norm(x_log[t, 0:2] - x_log[t, 7:9]).item()
        is_collision = dist < min_dist
        time_text.set_color("tab:red" if is_collision else "black")
        status = "collision" if is_collision else "clear"
        time_text.set_text(f"step {t}   distance {dist:.2f} m   {status} < {min_dist:.2f} m")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    update(0)
    plt.show()


if __name__ == "__main__":
    if EVALUATE_MODEL:
        evaluate_controller()
    show_simulation()
