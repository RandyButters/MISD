TIMESTEPS=$1
GPU=$2
SEED=$3

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
--env_name "Spiral-v0" \
--reward_shaping sparse \
--algo misd \
--version sparse \
--seed ${SEED} \
--max_timesteps ${TIMESTEPS} \
--no_correction \
--seed ${SEED}
