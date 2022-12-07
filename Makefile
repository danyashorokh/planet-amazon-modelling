
DVC_REMOTE_NAME := gstorage

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: download_weights
download_weights:
	dvc pull -R weights -r gstorage

.PHONY: run_tests
run_tests:
	PYTHONPATH=. pytest tests/

.PHONY: lint
lint:
	flake8 src/

.PHONY: init_dvc
init_dvc:
	dvc init --no-scm
	dvc remote add --default $(DVC_REMOTE_NAME) ssh://91.206.15.25/home/$(USERNAME)/dvc_files
	dvc remote modify $(DVC_REMOTE_NAME) user $(USERNAME)
	dvc config cache.type hardlink,symlink

.PHONY: install_c_libs
install_c_libs:
	apt-get update && apt-get install -y --no-install-recommends gcc ffmpeg libsm6 libxext6

simple_predict:
	python predict.py -w weights/model.best.zip -i  path/to/your/image --show
