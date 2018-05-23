SRCS = $(wildcard ./yapycrf/*.py ./yapycrf/*/*.py)


.PHONY : lint
lint :
	-pylint --rcfile=./.pylintrc ./yapycrf/*
	-pydocstyle --config=./.pydocstyle ./yapycrf/*
