OUT_FOLDER ?= $(MODEL)
HOSTPORT ?= 5001
DOCKER = docker
IMAGE = prosit
DOCKERFILE = Dockerfile


build:
	$(DOCKER) build -qf $(DOCKERFILE) -t $(IMAGE) .


server: build
	$(DOCKER) run -it \
	    -v "$(MODEL_CONFIG_SPECTRA)":/root/model_config_spectra/ \
	    -v "$(MODEL_CONFIG_RT)":/root/model_config_rt/ \
	    -v "$(WEIGHTS_CID)":/root/weights_cid/ \
	    -v "$(WEIGHTS_HCD)":/root/weights_hcd/ \
	    -v "$(WEIGHTS_RT)":/root/weights_rt/ \
	    -p $(HOSTPORT):$(HOSTPORT) \
	    $(IMAGE) python3 -m prosit.server -p $(HOSTPORT)

train: build
	$(DOCKER) run -it \
	    -v "$(TRAIN_DIR)":/root/training/ \
	    -e CUDA_VISIBLE_DEVICES=$(GPU) \
	    $(IMAGE) python3 -m prosit.train_prosit