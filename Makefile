.PHONY: .check_yesno owner-git owner-github clean conda-create-env conda-update-env install-dev-dependencies install-pre-commit uninstall-pre-commit install-package uninstall-package datasets

#################################################################################
# GLOBALS                                                                       #
#################################################################################

SHELL=/bin/bash

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = mpcstb
PACKAGE_NAME = mpcstb
ENV_NAME = landscape_steiner
SRC_CODE_FOLDER = src/mpcstb
RAW_DATA_OUTPUT_FOLDER = ${PROJECT_DIR}/data/raw
INTERM_DATASET_URL = https://github.com/larissaftf/IG-instances/raw/master/Interm.zip
PCSPG_PUCNU_DATASET_URL = http://dimacs11.zib.de/instances/PCSPG-PUCNU.zip
PYTHON_INTERPRETER = python
CURRENT_ENV := $(CONDA_DEFAULT_ENV)


ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
CONDA := $(shell which conda)
ifeq ($(CONDA_DEFAULT_ENV),$(ENV_NAME))
ENV_IS_ACTIVE=True
else
ENV_IS_ACTIVE=False
endif
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

# yes/no prompt
.check_yesno:
	@echo -n "If you cloned this repo, there is no need to do it. Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]

## Init version control. Do it only if you are the project owner
owner-git: .check_yesno
	@if [ ! -d ".git" ]; then \
	git init ; \
	versioneer install ; \
	git add . ; \
	git commit -m "INIT: Initial Commit" ; \
	fi

## Set github remote and push. Do it only if you are the project owner
owner-github: .check_yesno
	@if [ ! -d ".git" ]; then \
	make owner-git ; \
	fi

	@read -p "Enter remote repo https: " remote ; \
	git remote add origin $$remote ; \
	git push -u origin master ; \

## Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;

## create conda environment
conda-create-env:
ifeq (True,$(HAS_CONDA))
	@printf ">>> Creating '$(ENV_NAME)' conda environment. This could take a few minutes ...\n\n"
	@$(CONDA) env create --name $(ENV_NAME) --file environment.yml
	@$(CONDA) env export --name $(ENV_NAME) | grep -v -E -e '^\s*prefix:' > environment.lock.yml
	@printf ">>> Adding the project to the environment...\n\n"
else
	@printf ">>> conda command not found. Check out that conda has been installed properly."
endif

## update conda environment
conda-update-env:
ifeq (True,$(HAS_CONDA))
	@printf ">>> Updating '$(ENV_NAME)' conda environment. This could take a few minutes ...\n\n"
	@$(CONDA) env update --name $(ENV_NAME) --file environment.yml --prune
	@$(CONDA) env export --name $(ENV_NAME) | grep -v -E -e '^\s*prefix:' > environment.lock.yml
	@printf ">>> Updated.\n\n"
else
	@printf ">>> conda command not found. Check out that conda has been installed properly."
endif

## install develop dependencies
install-dev-dependencies:
	$(CONDA) run --name '$(ENV_NAME)' python -m pip install -U -r requirements-dev.txt

## Activate pre-commit
install-pre-commit:
	$(CONDA) run --name '$(ENV_NAME)' pre-commit install
	$(CONDA) run --name '$(ENV_NAME)' pre-commit install -t pre-commit
	$(CONDA) run --name '$(ENV_NAME)' pre-commit install -t pre-push

## Deactivate pre-commit
uninstall-pre-commit:
	$(CONDA) run --name '$(ENV_NAME)' pre-commit uninstall

## install package in editable mode
install-package:
	$(CONDA) run --name '$(ENV_NAME)' python -m pip install --editable .

## uninstall package
uninstall-package:
	$(CONDA) run --name '$(ENV_NAME)' python -m pip uninstall --yes '$(PACKAGE_NAME)'


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## download datasets
datasets:
	@printf ">>> Downloading `Interm` dataset."
	pushd ${RAW_DATA_OUTPUT_FOLDER} && \
		wget ${INTERM_DATASET_URL} \
		&& unzip Interm.zip \
		&& rm Interm.zip \
		&& popd

	@printf ">>> Downloading `PCSPG-PUCNU` dataset."
	pushd ${RAW_DATA_OUTPUT_FOLDER} && \
		wget ${PCSPG_PUCNU_DATASET_URL} \
		&& unzip PCSPG-PUCNU.zip \
		&& rm PCSPG-PUCNU.zip \
		&& popd

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
