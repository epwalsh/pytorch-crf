test = ./test/

SRCS = $(wildcard ./yapycrf/*.py ./yapycrf/*/*.py)

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
	@export PYTHONPATH=. && pytest $(test)
