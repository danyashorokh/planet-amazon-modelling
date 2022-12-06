
#install:
#	pip install -r requirements.txt

#download_weights:
#	dvc pull -R weights

#run_tests:
#	PYTHONPATH=. pytest tests/

lint:
	flake8 src/

#init_dvc:
#	dvc init --no-scm
#	dvc remote add --default $(DVC_REMOTE_NAME) ssh://91.206.15.25/home/$(USERNAME)/dvc_files
#	dvc remote modify $(DVC_REMOTE_NAME) user $(USERNAME)
#	dvc config cache.type hardlink,symlink
