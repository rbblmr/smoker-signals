SHELL:=/bin/bash

# Clone repo and move to the repo directory to use the make file
# git clone https://github.com/rbblmr/smoker-status.git
# sudo apt install make

build: 
	@docker build -t smoker-model .
run:
	@docker run -it --rm -p 9696:9696 smoker-model
clean:
	docker-compose down --volumes --rmi all

deploy:
	eb init -p docker -r ap-southeast-1 smoker-model-project
# eb local run --port 9696
	eb create smoker-serving-env
terminate-deploy:
	eb terminate smoker-serving-env