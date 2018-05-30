test = ./test/

PYTHON_VERSION := `grep "python:" Dockerfile | head -1 | sed -r 's/.*([0-9]\.[0-9]).*/\1/g'`
IMAGE_TAG      := python$(PYTHON_VERSION)


.PHONY : help
help :
	@echo \
		"Commands:\n"\
		"  lint: run pylint and pydocstyle on source files.\n"\
		"  test: run pytest on the test/ directory."

.PHONY : lint
lint :
	-@pydocstyle --config=./.pydocstyle ./yapycrf/*
	-@pylint --rcfile=./.pylintrc ./yapycrf/*

.PHONY : test
test :
	@export PYTHONPATH=. && pytest --cov=yapycrf $(test)

.PHONY : create-branch
create-branch :
	git checkout -b ISSUE-$(num)
	git push --set-upstream origin ISSUE-$(num)
