const SESSION_KEY_PREFIX = "chat_session_";

function sessionKey(repoId: string): string {
  return `${SESSION_KEY_PREFIX}${repoId}`;
}

function isStorageAvailable(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

export function getSession(repoId: string): string | null {
  if (!isStorageAvailable()) {
    return null;
  }

  try {
    return localStorage.getItem(sessionKey(repoId));
  } catch {
    return null;
  }
}

export function saveSession(repoId: string, sessionId: string): void {
  if (!isStorageAvailable()) {
    return;
  }

  try {
    localStorage.setItem(sessionKey(repoId), sessionId);
  } catch {
    // Ignore storage quota/security errors.
  }
}

export function clearSession(repoId: string): void {
  if (!isStorageAvailable()) {
    return;
  }

  try {
    localStorage.removeItem(sessionKey(repoId));
  } catch {
    // Ignore storage quota/security errors.
  }
}

export function hasSession(repoId: string): boolean {
  return getSession(repoId) !== null;
}

export function listActiveSessions(): string[] {
  if (!isStorageAvailable()) {
    return [];
  }

  try {
    const repoIds: string[] = [];
    for (let index = 0; index < localStorage.length; index += 1) {
      const key = localStorage.key(index);
      if (!key || !key.startsWith(SESSION_KEY_PREFIX)) {
        continue;
      }
      repoIds.push(key.slice(SESSION_KEY_PREFIX.length));
    }
    return repoIds;
  } catch {
    return [];
  }
}
