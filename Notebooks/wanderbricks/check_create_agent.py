try:
    from langchain.agents import create_agent
    import inspect
    print("Signature:", inspect.signature(create_agent))
    print("Docstring:", create_agent.__doc__)
except ImportError as e:
    print(e)
except Exception as e:
    print(e)
