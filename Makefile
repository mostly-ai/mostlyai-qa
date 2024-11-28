.PHONY: help
help: ## show definition of each function
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z1-9_-]+:.*?## / {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: clean
clean: ## remove .gitignore files
	git clean -fdX

.PHONY: install
install: ## install dependencies
	poetry install

.PHONY: lint
lint: ## run lints
	poetry run pre-commit run --all-files

.PHONY: test
test: ## run tests
	poetry run pytest

.PHONY: all
all: clean install lint test ## run all commands

.PHONY: examples
examples: ## run all examples
	find ./examples -maxdepth 1 -type f -name "*.ipynb" -print -execdir jupyter nbconvert --to script {} \;
	find ./examples -maxdepth 1 -type f -name "*.py" -print -execdir python {} \;

# Targets for Release Workflow/Automation
.PHONY: release-pypi bump-version update-vars-version build confirm-upload upload clean-dist docs

release-pypi: clean-dist build upload

# Default files to update
PYPROJECT_TOML = pyproject.toml
INIT_FILE = src/mostlyai/qa/__init__.py

# Default bump type
BUMP_TYPE ?= PATCH
CURRENT_VERSION := $(shell grep -m 1 'version = ' $(PYPROJECT_TOML) | sed -e 's/version = "\(.*\)"/\1/')
# Assuming current_version is already set from pyproject.toml
NEW_VERSION := $(shell echo $(CURRENT_VERSION) | awk -F. -v bump=$(BUMP_TYPE) '{ \
    if (bump == "PATCH") { \
        printf("%d.%d.%d", $$1, $$2, $$3 + 1); \
    } else if (bump == "MINOR") { \
        printf("%d.%d.0", $$1, $$2 + 1); \
    } else if (bump == "MAJOR") { \
        printf("%d.0.0", $$1 + 1); \
    } else { \
        print "Error: Invalid BUMP_TYPE=" bump; \
        exit 1; \
    } \
}')

# Rule to bump the version
bump-version:
	@echo "Bumping $(BUMP_TYPE) version from $(CURRENT_VERSION) to $(NEW_VERSION)"
	@echo "Replaces $(CURRENT_VERSION) to $(NEW_VERSION) in $(PYPROJECT_TOML)"
	@echo "Replaces $(CURRENT_VERSION) to $(NEW_VERSION) in $(INIT_FILE)"
	@echo "Current directory: $(shell pwd)"
    # Check if current version was found
	@if [ -z "$(CURRENT_VERSION)" ]; then \
        echo "Error: Could not find current version in $(PYPROJECT_TOML)"; \
        exit 1; \
    fi
    # Replace the version in pyproject.toml
	@if [[ "$(shell uname -s)" == "Darwin" ]]; then \
        sed -i '' 's/version = "$(CURRENT_VERSION)"/version = "$(NEW_VERSION)"/g' $(PYPROJECT_TOML); \
        sed -i '' 's/__version__ = "$(CURRENT_VERSION)"/__version__ = "$(NEW_VERSION)"/g' $(INIT_FILE); \
    else \
        sed -i 's/version = "$(CURRENT_VERSION)"/version = "$(NEW_VERSION)"/g' $(PYPROJECT_TOML); \
        sed -i 's/__version__ = "$(CURRENT_VERSION)"/__version__ = "$(NEW_VERSION)"/g' $(INIT_FILE); \
    fi
    @VERSION=$$(poetry version -s)
	@echo "Now we have version $(VERSION) in $(PYPROJECT_TOML) and $(INIT_FILE)."

update-vars-version: ## Update the required variables after bump
	$(eval VERSION := $(shell poetry version -s))
	$(eval BRANCH := verbump_$(shell echo $(VERSION) | tr '.' '_'))
	$(eval TAG := $(VERSION))
	@echo "Updated VERSION to $(VERSION), BRANCH to $(BRANCH), TAG to $(TAG)"

build: ## build package
	@echo "Step: building package"
	poetry build
	twine check --strict dist/*

confirm-upload: ## Confirm before the irreversible zone
	@echo "Are you sure you want to upload to PyPI? (yes/no)"
	@read ans && [ $${ans:-no} = yes ]

upload: confirm-upload ## Upload to PyPI (ensure the token is present in .pypirc file before running upload)
	@twine upload dist/*$(VERSION)* --verbose
	@echo "Uploaded version $(VERSION) to PyPI"

clean-dist: ## Remove "volatile" directory dist
	@echo "Step: cleaning dist directory"
	@rm -rf dist
	@echo "Cleaned up dist directory"
