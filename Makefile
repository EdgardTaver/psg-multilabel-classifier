## help display
## pulls comments from beside commands and prints a nicely formatted
## display with the commands and their usage information

.DEFAULT_GOAL := help

help: ## prints this help
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

list-requirements: ## lists requirements to a requirements file
	python3.8 -m pipreqs.pipreqs

list-requirements-force: ## lists requirements to a requirements file
	python3.8 -m pipreqs.pipreqs --force

install-requirements: ## install the requirements from the requirements file
	pip3.8 install --upgrade pip
	python3.8 -m pip install -r requirements.txt

test:
	python3.8 -m pytest

create-env: ## creates a virtual environment
	python3.8 -m venv env

activate-env: ## activates the virtual environment
	source env/bin/activate

check-env: ## checks if the virtual environment is active
	@echo "VIRTUAL_ENV: $(VIRTUAL_ENV)"