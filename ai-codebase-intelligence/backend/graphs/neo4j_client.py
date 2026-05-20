"""Compatibility shim exposing Neo4j helpers under graphs.* namespace."""

from graph.neo4j_client import (  # noqa: F401
	close_driver,
	get_database_stats,
	get_driver,
	get_session,
	ping,
	run_query,
)

__all__ = [
	"close_driver",
	"get_database_stats",
	"get_driver",
	"get_session",
	"ping",
	"run_query",
]
