from src.substitution.features.cleaners import clean_gender, clean_audience, clean_style, clasificar_binario


def test_clean_gender():
    assert clean_gender("masculino") == "HOMBRE"
    assert clean_gender("sin genero") == "UNISEX"


def test_clean_audience():
    assert clean_audience("teens") == "TEENS"
    assert clean_audience("preescolar") == "INFANTIL"


def test_clean_style():
    assert clean_style("travel") == "VIAJE"
    assert clean_style("moda") == "CASUAL"


def test_clasificar_binario():
    assert clasificar_binario("NO APLICA") == 0
    assert clasificar_binario("SI") == 1
