# CARLA path
export CARLA_ROOT=/home/kchitta/Documents/CARLA_0.9.10.1 # path to CARLA install
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0

# Server Ports
export PORT=2000 # same as the carla server port
export TM_PORT=8000 # port for traffic manager, required when spawning multiple servers/clients

# Evaluation Setup
export ROUTES=leaderboard/data/evaluation_routes/eval_routes_weathers.xml # routes file
export SCENARIOS=leaderboard/data/scenarios/eval_scenarios.json # scenarios file
export REPETITIONS=1 # number of evaluation runs
export RESUME=0 # resume evaluation from partial results file

# Agent Paths
export TEAM_AGENT=leaderboard/team_code/auto_pilot.py # agent script
export TEAM_CONFIG=./model_ckpt/neat # model checkpoint (not required for auto_pilot)
export CHECKPOINT_ENDPOINT=../carla_results/auto_pilot.json # output results file
# export SAVE_PATH=../carla_results/auto_pilot_eval # path for saving episodes (comment to disable)

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
