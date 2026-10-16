class AudioMixerError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class ValidationError(AudioMixerError):
    def __init__(self, message: str):
        super().__init__("validation_error", message, 422)


class NotFoundError(AudioMixerError):
    def __init__(self, code: str, message: str):
        super().__init__(code, message, 404)


class UnauthorizedError(AudioMixerError):
    def __init__(self):
        super().__init__("unauthorized", "API key is required or invalid", 401)


class ForbiddenError(AudioMixerError):
    def __init__(self):
        super().__init__("forbidden", "Required scope is missing", 403)


class ProcessingError(AudioMixerError):
    def __init__(self, message: str):
        super().__init__("processing_failed", message, 422)
