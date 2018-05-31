test = ./test/

PYTHON_VERSION := `grep "python:" Dockerfile | head -1 | sed -r 's/.*([0-9]\.[0-9]).*/\1/g'`
IMAGE_TAG      := python$(PYTHON_VERSION)


.PHONY : help
help :
	@echo \
		"Commands:\n"\
		"  lint:      runs pylint and pydocstyle on source files.\n"\
		"  typecheck: runs mypy to statically check types.\n"\
		"  test:      runs pytest on the test/ directory."

.PHONY : typecheck
typecheck :
	-@mypy pycrf --ignore-missing-imports

.PHONY : lint
lint :
	@echo "Lint (pydocstyle):\n"
	-@pydocstyle --config=./.pydocstyle pycrf
	@echo "\nLint (pylint):\n"
	-@pylint --rcfile=./.pylintrc -f colorized pycrf

.PHONY : test
test :
	@echo "Unit tests (pytest):"
	@export PYTHONPATH=. && pytest --color=yes -v --cov=pycrf $(test)

.PHONY : create-branch
create-branch :
	git checkout -b ISSUE-$(num)
	git push --set-upstream origin ISSUE-$(num)
