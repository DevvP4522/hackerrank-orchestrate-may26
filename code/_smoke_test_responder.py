import importlib.util

spec = importlib.util.spec_from_file_location(
    "responder",
    "hackerrank-orchestrate-may26/code/responder.py",
)
m = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m)

issue = "I can't reset my password"
retrieved_docs = [
    {
        "text": "To reset your password, go to the login page and click 'Forgot Password'. Follow the instructions sent to your email.",
        "source": "doc",
        "product_area": "login",
        "score": 0.9,
    }
]

print(m.generate_response(issue, retrieved_docs))
