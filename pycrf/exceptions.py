"""Custom exceptions."""

from typing import List
import re


class Error(Exception):
    """Base exception."""

    pass


class LearnerInitializationError(Error):
    """Bad arguments passed to initialize learner object."""

    def __init__(self,
                 missing_args: List[str] = None,
                 unknown_args: List[str] = None) -> None:
        message: str = None
        self.missing_args = missing_args
        self.unknown_args = unknown_args
        if unknown_args:
            message = f"unrecognized keyword argument: "\
                f"{', '.join([x.replace('-', '_') for x in unknown_args])}"
        elif missing_args:
            message = f"missing required keyword argument: "\
                f"{', '.join([x.replace('-', '_') for x in missing_args])}"
        super(LearnerInitializationError, self).__init__(message)


class ArgParsingError(Error):
    """Bad command-line options."""

    re_args = re.compile(r"--([a-zA-Z0-9-]+)")

    def __init__(self, message: str = None) -> None:
        self.message = message
        self.missing_args: List[str] = []
        self.unknown_args: List[str] = []
        if message and "the following arguments are required" in message:
            self.missing_args = self.re_args.findall(message)
        elif message and "unrecognized arguments" in message:
            self.unknown_args = self.re_args.findall(message)
        super(ArgParsingError, self).__init__(message)
