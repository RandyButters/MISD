TIMESTEPS=$1
GPU=$2
SEED=$3

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
--env_name "AntTwoRooms-v2" \
--reward_shaping dense \
--algo misd \
--version dense \
--seed ${SEED} \
--max_timesteps ${TIMESTEPS} \
--no_correction \
--seed ${SEED}