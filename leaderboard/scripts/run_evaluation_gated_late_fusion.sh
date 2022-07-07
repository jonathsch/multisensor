#!/bin/bash

# ------------------------------------------------------------------------
# --------------------- 1. regular evaluation ----------------------------
# ------------------------------------------------------------------------

export CARLA_ROOT=/storage/group/intellisys/CARLA/CARLA_9.10
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000 # same as the carla server port
export TM_PORT=8000 # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export REPETITIONS=3 # multiple evaluation runs
export ROUTES=leaderboard/data/evaluation_routes/routes_town05_long.xml
export TEAM_AGENT=leaderboard/team_code/gated_late_fusion_agent.py # agent
export TEAM_CONFIG=log/gated_late_fusion_final # model checkpoint, not required for expert
export CHECKPOINT_ENDPOINT=results/gated_late_fusion_final/regular.json # results file
export SCENARIOS=leaderboard/data/scenarios/town05_all_scenarios.json
export SAVE_PATH=data/gated_late_fusion_final # path for saving episodes while evaluating
export RESUME=False

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}

# ------------------------------------------------------------------------
# --------------------- 1. no lidar evaluation ---------------------------
# ------------------------------------------------------------------------

export TEAM_AGENT=leaderboard/team_code/gated_late_fusion_agent_nolidar.py # agent
export TEAM_CONFIG=log/gated_late_fusion_final # model checkpoint, not required for expert
export CHECKPOINT_ENDPOINT=results/gated_late_fusion_final/no_lidar.json # results file
export SAVE_PATH=data/gated_late_fusion_final_no_lidar # path for saving episodes while evaluating

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}

# ------------------------------------------------------------------------
# --------------------- 1. no rgb evaluation ---------------------------
# ------------------------------------------------------------------------

export TEAM_AGENT=leaderboard/team_code/gated_late_fusion_agent_norgb.py # agent
export TEAM_CONFIG=log/gated_late_fusion_final # model checkpoint, not required for expert
export CHECKPOINT_ENDPOINT=results/gated_late_fusion_final/no_rgb.json # results file
export SAVE_PATH=data/gated_late_fusion_final_no_rgb # path for saving episodes while evaluating

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}