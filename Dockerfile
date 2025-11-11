# base image (lightweight python runtime)
FROM python:3.11-slim

# set working directory
WORKDIR /app

# force to install cpu version of torch
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install lightning

# copy and install dependency file (with torch cpu version)
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy training script
COPY training.py .

# create output directory for model checkpoints
RUN mkdir /app/models

# default command to run training with best hyperparameters
CMD [ \
    "python", "training.py", \
    "--checkpoint_dir", "/app/models", \
    "--learning_rate", "2.642988043609753e-05", \
    "--train_batch_size", "64", \
    "--eval_batch_size", "32", \
    "--weight_decay", "0.032716372623540435", \
    "--warmup_ratio", "0.1", \
    "--beta1", "0.9" \
]