import os

# load env variables from .env file (<KEY>=<VAL>)
def load_env_file(filepath):
	with open(filepath) as f:
		for line in f:
			key, value = line.strip().split("=", 1)
			os.environ[key] = value