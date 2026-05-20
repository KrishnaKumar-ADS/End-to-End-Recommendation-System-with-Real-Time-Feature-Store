import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  clearSession,
  getSession,
  hasSession,
  listActiveSessions,
  saveSession,
} from "@/lib/session";

describe("session", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("returns null when session is missing", () => {
    expect(getSession("repo-1")).toBeNull();
  });

  it("saves and retrieves a session id", () => {
    saveSession("repo-1", "session-abc");
    expect(getSession("repo-1")).toBe("session-abc");
  });

  it("overwrites a previous session id", () => {
    saveSession("repo-1", "session-old");
    saveSession("repo-1", "session-new");

    expect(getSession("repo-1")).toBe("session-new");
  });

  it("clears a stored session", () => {
    saveSession("repo-1", "session-abc");
    clearSession("repo-1");

    expect(getSession("repo-1")).toBeNull();
    expect(hasSession("repo-1")).toBe(false);
  });

  it("reports hasSession for stored sessions", () => {
    saveSession("repo-1", "session-abc");
    expect(hasSession("repo-1")).toBe(true);
  });

  it("lists repo ids that have active sessions", () => {
    saveSession("repo-1", "s1");
    saveSession("repo-2", "s2");
    localStorage.setItem("unrelated", "value");

    expect(listActiveSessions().sort()).toEqual(["repo-1", "repo-2"]);
  });

  it("returns null from getSession when storage throws", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });

    expect(getSession("repo-1")).toBeNull();
  });

  it("returns empty array from listActiveSessions when storage throws", () => {
    vi.spyOn(Storage.prototype, "key").mockImplementation(() => {
      throw new Error("blocked");
    });

    expect(listActiveSessions()).toEqual([]);
  });
});
