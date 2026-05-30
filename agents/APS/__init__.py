"""
agents/APS

Schema Matcher Agent. Especializaciones por motor en mysql/ y postgres/.

Uso:
    from agents.APS import SchemaMatcherAgent
    agent = SchemaMatcherAgent(db_type="mysql")    # -> MysqlSchemaMatcherAgent
    agent = SchemaMatcherAgent(db_type="postgres") # -> PostgresSchemaMatcherAgent
"""


def SchemaMatcherAgent(schema_path=None, vector_db_path=None, db_type: str = "mysql",
                       top_n_tables: int = 5, top_n_columns: int = 10):
    """
    Factory que devuelve el agente APS especializado segun el motor de BD.
    La firma es identica al constructor de las subclases.
    """
    motor = (db_type or "mysql").lower()
    kwargs = {"top_n_tables": top_n_tables, "top_n_columns": top_n_columns}
    if schema_path is not None:
        kwargs["schema_path"] = schema_path
    if vector_db_path is not None:
        kwargs["vector_db_path"] = vector_db_path

    if motor == "postgres":
        from agents.APS.postgres.schema_matcher_agent import PostgresSchemaMatcherAgent
        return PostgresSchemaMatcherAgent(**kwargs)
    else:
        from agents.APS.mysql.schema_matcher_agent import MysqlSchemaMatcherAgent
        return MysqlSchemaMatcherAgent(**kwargs)


__all__ = ["SchemaMatcherAgent"]
