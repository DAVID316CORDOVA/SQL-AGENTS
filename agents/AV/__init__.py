"""
agents/AV

SQL Validator Agent. Especializaciones por motor en mysql/ y postgres/.

Uso:
    from agents.AV import SQLValidatorAgent
    agent = SQLValidatorAgent(db_type="mysql")
    agent = SQLValidatorAgent(db_type="postgres")
"""


def SQLValidatorAgent(db_type: str = "mysql"):
    motor = (db_type or "mysql").lower()
    if motor == "postgres":
        from agents.AV.postgres.sql_validator_agent import SQLValidatorAgent as _PG
        return _PG()
    else:
        from agents.AV.mysql.sql_validator_agent import SQLValidatorAgent as _MY
        return _MY()


__all__ = ["SQLValidatorAgent"]
