from src.tonality import tonality

"""neutral"""

text = """
большие емкости для перевозки жидкостей нефти и продуктов ее переработки часто используются в виде вагонов поездов
"""


def test_neutral():
    print("Test 3 - ", tonality(text))
    return tonality(text)