SHELL:=/bin/bash

# Clone repo and move to the repo directory to use the make file
# git clone https://github.com/rbblmr/smoker-signals.git
# sudo apt install make

build: 
	docker build -t smoker-model .
run:
	docker run -it --rm -p 9696:9696 smoker-model
test-local:
	python predict-test-local.py

deploy:
	eb init -p docker -r ap-southeast-1 smoker-model-project
# eb local run --port 9696
	eb create smoker-serving-env
test-cloud:
	python predict-test-cloud.py
terminate-app:
	eb terminate smoker-serving-env