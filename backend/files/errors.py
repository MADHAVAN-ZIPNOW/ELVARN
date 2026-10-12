"""Errors that carry a friendly message the user can read as-is."""


class FileError(Exception):
    status = 400

    def __init__(self, message, status=None):
        super().__init__(message)
        self.message = message
        if status is not None:
            self.status = status


class TooLarge(FileError):
    status = 413


class Unsupported(FileError):
    status = 415


class NotFound(FileError):
    status = 404


class Busy(FileError):
    status = 503


class LimitReached(FileError):
    status = 429
