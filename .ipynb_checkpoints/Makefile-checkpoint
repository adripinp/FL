#### Linux/apt manager only #### 
.DEFAULT_GOAL := run 

PYTHON = ./.env/.venv/bin/python3
PIP = ./.env/.venv/bin/pip

# Only run this if 
install:
	sudo apt install python3.10
	sudo apt install python3.10-venv
	sudo apt install python3.10-dev
	sudo apt-get install python3-tk

setup: python_requirements.txt
	python3.10 -m venv ./.env/.venv
	chmod +x ./.env/.venv/bin/activate
	. ./.env/.venv/bin/activate
	mkdir ./.env/.saved
	$(PIP) install -r python_requirements.txt

activate:
	. ./.env/.venv/bin/activate

# Runs environment.
run: activate
	$(PYTHON) main.py

# Clear saved client weights. 
clear:
	rm -rf ./.env/.saved			

# Deletes all installed dependencies. 
clean: clear
	rm -rf __pycache__
	rm -rf ./.env