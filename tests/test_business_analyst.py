from agents.business_analyst import business_analyst_agent


def test_business_analyst_agent_configuration():
    assert business_analyst_agent.name == "business_analyst"
    assert business_analyst_agent.description
    assert business_analyst_agent.instruction
