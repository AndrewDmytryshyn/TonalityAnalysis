from src.tonality import tonality

"""positive"""

text = """
всем добра и сгущенки
"""


def test_positive():
    print("Test 1 - ", tonality(text))
    return tonality(text)