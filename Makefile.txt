PYTHON ?= python
CONFIG ?= configs/train_4gb.yaml

.PHONY: all prepare build train resume verify export push clean

all: prepare build train export

prepare:
	$(PYTHON) scripts/01_prepare_datasets.py

build:
	$(PYTHON) scripts/02_build_hf_dataset.py

train:
	$(PYTHON) scripts/03_train.py --config $(CONFIG)

resume:
	$(PYTHON) scripts/03_train.py --config $(CONFIG) --resume auto

verify:
	$(PYTHON) scripts/06_verify_resume.py

export:
	$(PYTHON) scripts/04_export_model.py

push:
	$(PYTHON) scripts/05_push_to_hub.py --repo_id $(REPO_ID)

clean:
	rm -rf data/t5_sql_dataset
	rm -rf models/t5_sql_finetuned