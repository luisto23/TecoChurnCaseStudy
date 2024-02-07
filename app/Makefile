.PHONY: setup_env remove_env data 
PROJECT_NAME=casestudy-env

ifeq (,$(shell which pyenv))
	HAS_PYENV=False
	CONDA_ROOT=$(shell conda info --root)
	BINARIES = ${CONDA_ROOT}/envs/${PROJECT_NAME}/bin
else
	HAS_PYENV=True
	CONDA_VERSION=$(shell echo $(shell pyenv version | awk '{print $$1;}') | awk -F "/" '{print $$1}')
	BINARIES = $(HOME)/.pyenv/versions/${CONDA_VERSION}/envs/${PROJECT_NAME}/bin
endif
data:
	@echo "Creating dataset.."
	${BINARIES}/python -m src.data.make_dataset


features:
	@echo "Running feature engineering on dataset.."
	${BINARIES}/python -m src.features.build_features

train:
	@echo "Training classification model for allocation task.."
	${BINARIES}/python -m src.models.train_model

predict:
	@echo "Performing model inference to churn customers.."
	${BINARIES}/python -m src.models.predict_model