Elastic Beanstalk

1. Start at the project directory (must have the dockerfile and pipenv file)
2. pipenv shell --> pipenv install awsebcli --dev
3. eb init -p docker -r `region` `name for eb project directry`
4. eb local run --port 9696 
    - port 9696 is the exposed docker port , 80 is the port of elastic beanstalk server
    - this will only run locally

5. eb create churn-serving-env
    - this will deploy the app using eb, like streamlit
    - the app will be available using an address - this address is automatically routed to a port (80)
    - change the host address with the address for eb server in your predict.py script
    - make sure you don't open this server to the public
        - make it available to a certain network only (specific client servers)

6. eb terminate churn-serving-env
    - shut down the server

Docker

Without using dockerfile:
- docker run -it --rm --entrypoint=bash `python and linux version` (manually configuring the entrypoint)
- apt-get install wget
- pip install pipenv
- mkdir app/
- cd app/
- cp pipfile and pipfile.lock to ./ (app/)
- cp predict.py, model to ./ (app/)
- gunicorn --bind=0.0.0.0:9696
- expose the port of gunicorn/docker to the host machine

- These are the steps you have to manually do without dockerfile

With dockerfile (dockerizing the steps):
- docker build -t name .
    ```dockerfile
    FROM python:3.7.5-slim

    ENV PYTHONUNBUFFERED=TRUE

    RUN pip --no-cache-dir install pipenv

    WORKDIR /app

    COPY ["Pipfile", "Pipfile.lock", "./"]
    RUN pipenv install --deploy --system && \
        rm -rf /root/.cache

    COPY ["*.py", "churn-model.bin", "./"]

    EXPOSE 9696

    ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "churn_serving:app"] 
    ```

    - we exposed the docker port to the host machine but we haven't mapped the 2 yet
    - docker run -it --rm -p 9696:9696 `name of docker container` (9696:9696 --> docker port:host port)

MLflow
- Probelm with setting up artifacts store
    - To setup a localhost directory as artifcats store in google colab
        - localhost must be setup as an sftp server
        - run ssh
        - Problem: SSH localhost won't run in gitbash
            - Solution: Run ssh using wsl (UBUNTU)
                - Download WSL from Microsoft Store
                - Download a distro 
                - Run UBUNTU
                    -       sudo apt-get upgrade
                            sudo apt-get update
                            sudo apt-get install openssh-server
                            sudo service ssh start

                            cd ~/.ssh
                            ssh-keygen to generate a public/private rsa key pair; use the default options
                            cat id_rsa.pub >> authorized_keys to append the key to the authorized_keys file
                            chmod 640 authorized_keys to set restricted permissions
                            sudo service ssh restart to pickup recent changes
                            ssh localhost

        -  sftp://user@host/path/to/directory
            - sftp://rbbelunix@LAPTOP-U7AJRIIT/mnt/c/Users/rbbel/OneDrive/ML/Midterms/mlruns
        - mlflow sever --backend-store-uri sqlite:///mlflow.db --default-artifact-root sftp://rbbelunix@LAPTOP-U7AJRIIT/mnt/c/Users/rbbel/OneDrive/ML/Midterms/mlruns
            - sftp must be configured in the notebook and in the pipenv
            - import pysftp after pip install pysftp
            - setup connection
            - ssh/known_hosts must be present in colab
                - copy the entry in known_hosts when you ssh localhost in your local terminal
            - the ssh server must be port forwarded to be public
            - Problem: sftp can't still be accessed
            - might need to configure ssh keys in colab or configure password


![](2023-11-04-19-30-32.png)
![](2023-11-04-19-31-10.png)
![](2023-11-04-19-55-13.png)
![Alt text](image-1.png)
![Alt text](image.png)