import pytest

from app.agents.router import route
from app.agents.tools import calculator, sql_query


def test_router_choices():
    assert route("amount=900 hour=2 merchant_risk=0.9").tool == "predict_fraud"
    assert route("what is 2 + 2").tool == "calculator"
    assert route("how many events were logged").tool == "sql_query"
    assert route("explain the vector store").tool == "search_docs"


def test_calculator_safe():
    assert calculator("2^3 + 1")["value"] == 9
    with pytest.raises(ValueError):
        calculator("__import__('os').system('ls')")


def test_sql_is_read_only():
    with pytest.raises(ValueError):
        sql_query("DROP TABLE event_logs")
    with pytest.raises(ValueError):
        sql_query("SELECT 1; DELETE FROM event_logs")
