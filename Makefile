test = ./test/
logdir = s3://structurely-ml-logs/pycrf/
port = 6006

PYTHON_VERSION := `grep "python:" Dockerfile | head -1 | sed -r 's/.*([0-9]\.[0-9]).*/\1/g'`
IMAGE_TAG      := python$(PYTHON_VERSION)


.PHONY : help
help :
	@echo \
		"Commands:\n"\
		"  lint:      runs pylint and pydocstyle on source files.\n"\
		"  typecheck: runs mypy to statically check types.\n"\
		"  unit-test: runs pytest on the test/ directory.\n"\
		"  test:      runs all of the above."

.PHONY : typecheck
typecheck :
	@echo "Typechecks (mypy):"
	-@mypy pycrf --ignore-missing-imports

.PHONY : lint
lint :
	@echo "Lint (pydocstyle):"
	-@pydocstyle --config=./.pydocstyle pycrf
	@echo "Lint (pylint):"
	-@pylint --rcfile=./.pylintrc -f colorized pycrf

.PHONY : unit-test
unit-test :
	@echo "Unit tests (pytest):"
	@export PYTHONPATH=. && pytest --color=yes -v --cov=pycrf $(test)

.PHONY : test
test : typecheck lint unit-test

.PHONY : create-branch
create-branch :
	git checkout -b ISSUE-$(num)
	git push --set-upstream origin ISSUE-$(num)

.PHONY : tensor-board
tensor-board :
	@google-chrome http://localhost:$(port)
	S3_REGION=us-west-2 TF_CPP_MIN_LOG_LEVEL=2 tensorboard \
		--logdir=$(logdir) \
		--port=$(port)

.PHONY : clean
clean :
	@find ./training/logs -type f | grep -E "tfevents" | xargs rm -f
