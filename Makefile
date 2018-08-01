test = ./test/
logdir = ./training/logs/

S3_LOGDIR = s3://structurely-ml-logs/pycrf/
TB_PORT   = 6006
REPO      = epwalsh/pytorch-crf


.PHONY : help
help :
	@echo \
		"Commands:\n"\
		"  lint:        run pylint and pydocstyle on source files.\n"\
		"  typecheck:   run mypy to statically check types.\n"\
		"  unit-test:   run pytest on the test/ directory.\n"\
		"  test:        run all of the above.\n"\
		"  pull-logs:   pull training logs from S3.\n"\
		"  tensorboard: start a tensorboard instance to examine logs."

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

.PHONY : pull-logs
pull-logs :
	@mkdir -p ./training/logs
	@aws s3 cp --recursive $(S3_LOGDIR) ./training/logs/

.PHONY : tensorboard
tensorboard :
	@if [ -x "$$(command -v google-chrome)" ]; then \
		google-chrome http://localhost:$(TB_PORT); \
	else \
		open http://localhost:$(TB_PORT); \
	fi
	@S3_REGION=us-west-2 TF_CPP_MIN_LOG_LEVEL=2 tensorboard \
		--logdir=$(logdir) \
		--port=$(TB_PORT)

.PHONY : docker-build
docker-build :
	docker build -t $(REPO) .

.PHONY : docker-test
docker-test :
	docker run --rm -it $(REPO)

.PHONY : clean
clean :
	@find ./training/logs -type f | grep -E "tfevents" | xargs rm -f
