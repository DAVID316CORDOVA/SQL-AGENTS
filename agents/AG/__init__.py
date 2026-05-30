"""
agents/AG

SQL Generator Agent. Especializaciones por motor en mysql/ y postgres/.

Uso:
    from agents.AG import SQLGeneratorAgent
    agent = SQLGeneratorAgent(db_type="mysql")
    agent = SQLGeneratorAgent(db_type="postgres")
"""


def SQLGeneratorAgent(db_type: str = "mysql"):
    motor = (db_type or "mysql").lower()
    if motor == "postgres":
        from agents.AG.postgres.sql_generator_agent import SQLGeneratorAgent as _PG
        return _PG()
    else:
        from agents.AG.mysql.sql_generator_agent import SQLGeneratorAgent as _MY
        return _MY()


__all__ = ["SQLGeneratorAgent"]
