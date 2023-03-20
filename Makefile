.PHONY: lint
lint:
	poetry run pre-commit run --all-files

.PHONY: install
install:
	poetry install

.PHONY: install-pre-commit
update-pre-commit:
	poetry run pre-commit uninstall
	poetry run pre-commit install


.PHONY: run-rank
run-rank:
	python -m main

.PHONY: update
update:	install	install-pre-commit
