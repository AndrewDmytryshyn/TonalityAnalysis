from src.tonality import tonality

"""negative"""

text = """
коллеги сидят рубятся в кс а я из-за долбанной винды не могу
"""


def test_negative():
    print("Test 2 - ", tonality(text))
    return tonality(text)