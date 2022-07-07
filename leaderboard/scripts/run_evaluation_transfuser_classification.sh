#!/bin/bash

# ------------------------------------------------------------------------
# --------------------- 1. regular  evaluation ---------------------------
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
export REPETITIONS=1 # multiple evaluation runs
export ROUTES=leaderboard/data/evaluation_routes/routes_town05_long.xml
export TEAM_AGENT=leaderboard/team_code/transfuser_agent_classification_branch.py # agent
export TEAM_CONFIG=log/transfuser_classification_branch
export CHECKPOINT_ENDPOINT=results/transfuser_classification_branch/regular.json # results file
export SCENARIOS=leaderboard/data/scenarios/town05_all_scenarios.json
export SAVE_PATH=data/transfuser_classification_branch # path for saving episodes while evaluating
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

export TEAM_AGENT=leaderboard/team_code/transfuser_agent_classification_branch_nolidar.py # agent
export TEAM_CONFIG=log/transfuser_classification_branch
export CHECKPOINT_ENDPOINT=results/transfuser_classification_branch/nolidar.json # results file
export SCENARIOS=leaderboard/data/scenarios/town05_all_scenarios.json
export SAVE_PATH=data/transfuser_classification_branch # path for saving episodes while evaluating

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
# --------------------- 1.   no rgb evaluation ---------------------------

export TEAM_AGENT=leaderboard/team_code/transfuser_agent_classification_branch_norgb.py # agent
export TEAM_CONFIG=log/transfuser_classification_branch
export CHECKPOINT_ENDPOINT=results/transfuser_classification_branch/norgb.json # results file
export SCENARIOS=leaderboard/data/scenarios/town05_all_scenarios.json
export SAVE_PATH=data/transfuser_classification_branch # path for saving episodes while evaluating

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