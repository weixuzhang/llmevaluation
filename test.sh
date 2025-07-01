# Create a safer test config that only uses dummy models
python -c "
import json
config = {
    'workspace_config': {'sink': 'dummy', 'project': 'test', 'log_folder': 'test_logs'},
    'task_config': {'task': 'summarization', 'num_train_ex': 2, 'seed': 0},
    'user_config': {'model': 'dummy'}, 
    'agent_configs': {'no_learning': {'model': 'dummy', 'agent': 'no-learning'}}
}
with open('test_config.json', 'w') as f:
    json.dump(config, f, indent=2)
"

# Run the test with PYTHONPATH set
PYTHONPATH=. python src/main.py --experiments_config test_config.json