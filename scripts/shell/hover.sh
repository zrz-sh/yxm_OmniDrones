# total_frames=250_000_000
# algorithm="sac"  # mappo, maddpg, dqn, sac, td3
# action_transform=""  # PIDrate, null
# throttles_in_obs=true # true, false
# wandb_project="omnidrones"
# seed=0  # 0, 1, 2
# cd "$(dirname "$0")"
# CUDA_VISIBLE_DEVICES=0 python train.py headless=true \
#     total_frames=${total_frames} \
#     task=SingleJuggleVolleyball \
#     task.drone_model=Iris \
#     task.env.num_envs=2048 \
#     task.ball_mass=0.005 \
#     task.ball_radius=0.1 \
#     eval_interval=50 \
#     save_interval=500 \
#     algo=${algorithm} \
#     algo.buffer_size=1_000_000 \
#     algo.batch_size=4096 \
#     algo.gradient_steps=512 \
#     task.time_encoding=false \
#     task.action_transform=${action_transform}\
#     task.throttles_in_obs=${throttles_in_obs}\
#     seed=${seed} \
#     # wandb.mode=disabled \
#     # wandb.project=${wandb_project} \


total_frames=200_000_000 # 20_000_000 for results, 20_000_000 for debug
algorithm="sac"  # mappo, matd3 (maddpg), happo, mat
action_transform=""  # PIDrate, null
wandb_project="omnidrones"
seed=0  # 0, 1, 2

# export WANDB_API_KEY='853ccd594b6cb6b8955900d478f3f3384f64383d'
CUDA_VISIBLE_DEVICES=0 python train.py headless=true \
    total_frames=${total_frames} \
    task=Hover \
    task.drone_model=Iris \
    task.env.num_envs=4096 \
    eval_interval=50 \
    save_interval=500 \
    algo=${algorithm} \
    algo.buffer_size=1_000_000 \
    algo.batch_size=4096 \
    algo.gradient_steps=512 \
    task.time_encoding=false \
    seed=${seed} \
    # wandb.mode=disabled \

