FROM continuumio/miniconda3:latest

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate cli-chatbot" > /etc/profile.d/conda.sh
ENV PATH /opt/conda/envs/cli-chatbot/bin:$PATH

WORKDIR /app
COPY . /app

CMD ["python", "chatbot.py", "chat"]
