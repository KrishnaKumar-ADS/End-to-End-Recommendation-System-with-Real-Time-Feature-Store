class AppError(Exception):
    """Base application error."""
    pass

class RepoNotFoundError(AppError):
    """Repository could not be cloned or does not exist."""
    pass

class RepoAlreadyIndexedError(AppError):
    """Repository has already been ingested."""
    pass

class TaskNotFoundError(AppError):
    """Celery task ID not found."""
    pass

class UnsupportedLanguageError(AppError):
    """File language is not supported for parsing."""
    pass

class LLMProviderError(AppError):
    """All LLM providers failed to respond."""
    pass

class EmbeddingError(AppError):
    """Embedding generation failed."""
    pass


class ChunkNotFoundError(AppError):
    """Requested code chunk could not be found."""
    pass