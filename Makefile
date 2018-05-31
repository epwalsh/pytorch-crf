test = ./test/

PYTHON_VERSION := `grep "python:" Dockerfile | head -1 | sed -r 's/.*([0-9]\.[0-9]).*/\1/g'`
IMAGE_TAG      := python$(PYTHON_VERSION)


.PHONY : help
help :
	@echo \
		"Commands:\n"\
		"  lint: run pylint and pydocstyle on source files.\n"\
		"  test: run pytest on the test/ directory."

.PHONY : typecheck
typecheck :
	-@mypy ./pycrf/ --ignore-missing-imports

.PHONY : lint
lint :
	-@pydocstyle --config=./.pydocstyle ./pycrf/*
	-@pylint --rcfile=./.pylintrc ./pycrf/*

.PHONY : test
test :
	@export PYTHONPATH=. && pytest --cov=pycrf $(test)

.PHONY : create-branch
create-branch :
	git checkout -b ISSUE-$(num)
	git push --set-upstream origin ISSUE-$(num)
