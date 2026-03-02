import pandas as pd
import pytest

from hedonic_analysis.data_management.clean_housing import (
    _AMENITY_PATTERNS,
    _classify_category,
    _create_count_dummies,
    _detect_offplan,
    _detect_outliers,
    _extract_amenities,
    _extract_bairro,
    _extract_price,
    _fail_if_bairro_not_valid,
    _fail_if_price_negative,
    clean_housing,
)

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def raw_price_series():
    return pd.Series(
        [
            "vendaR$ 2.300.000Me avisar se o preço baixar",
            "R$ 450.000",
            None,
            "vendaR$ 1.200.000",
        ],
        name="Preco",
    )


@pytest.fixture
def raw_df_minimal():
    """Minimal DataFrame that exercises the full pipeline."""
    return pd.DataFrame(
        {
            "URL": [
                "https://www.imovelweb.com.br/apartamentos/x",
                "https://www.imovelweb.com.br/casas/y",
                "https://www.imovelweb.com.br/apartamentos/z",
            ],
            "Endereco": [
                "R. Exemplo 123",
                "Avenida Teste, 456",
                "Al. Outra 789",
            ],
            "Bairro": [
                "Curitiba, Paraná, Brasil, ",
                "Curitiba, Paraná, Brasil, ",
                "Curitiba, Paraná, Brasil, ",
            ],
            "Tipo": [None, None, None],
            "Categoria": [None, None, None],
            "Area_total_m2": [120.0, 200.0, 80.0],
            "Area_util_m2": [100.0, 180.0, 65.0],
            "N_quartos": [3.0, 4.0, 2.0],
            "N_banheiros": [2.0, 3.0, 1.0],
            "N_vagas": [2.0, 3.0, 1.0],
            "Preco": [
                "vendaR$ 800.000Me avisar",
                "vendaR$ 1.500.000Me avisar",
                "vendaR$ 350.000Me avisar",
            ],
            "IPTU": [None, None, None],
            "Idade_anos": ["10\n\t\t anos", "Breve Lançamento", "5\n\t\t anos"],
            "Adicionais": [None, None, None],
            "Areas_comuns": [None, None, None],
            "Areas_privativas": [None, None, None],
            "Descricao": [
                "Lindo apto com piscina e academia Água Verde",
                "Casa com churrasqueira e playground Batel",
                "Apartamento com elevador Centro",
            ],
            "Planta": [None, None, None],
        },
    )


# ------------------------------------------------------------------ #
# _extract_price
# ------------------------------------------------------------------ #


def test_extract_price_venda_format(raw_price_series):
    got = _extract_price(raw_price_series)
    assert got.iloc[0] == pytest.approx(2_300_000.0)


def test_extract_price_simple_format(raw_price_series):
    got = _extract_price(raw_price_series)
    assert got.iloc[1] == pytest.approx(450_000.0)


def test_extract_price_null_stays_null(raw_price_series):
    got = _extract_price(raw_price_series)
    assert pd.isna(got.iloc[2])


def test_extract_price_dtype_is_float32(raw_price_series):
    got = _extract_price(raw_price_series)
    assert got.dtype == pd.Float32Dtype()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("vendaR$ 100.000Me avisar", 100_000.0),
        ("R$ 50.000", 50_000.0),
        ("R$ 10.500.000", 10_500_000.0),
        ("aluguelR$ 3.200Me avisar", 3_200.0),
    ],
)
def test_extract_price_various_formats(text, expected):
    series = pd.Series([text], name="Preco")
    got = _extract_price(series)
    assert got.iloc[0] == pytest.approx(expected)


# ------------------------------------------------------------------ #
# _extract_bairro
# ------------------------------------------------------------------ #


def test_extract_bairro_simple_tail():
    df = pd.DataFrame(
        {"Descricao": ["Apartamento bonito Água Verde"]},
    )
    got = _extract_bairro(df)
    assert got.iloc[0] == "Água Verde"


def test_extract_bairro_ascii_tail():
    df = pd.DataFrame(
        {"Descricao": ["Apartamento bonito Agua Verde"]},
    )
    got = _extract_bairro(df)
    assert got.iloc[0] == "Água Verde"


def test_extract_bairro_correction_applied():
    df = pd.DataFrame(
        {"Descricao": ["Apartamento novo Ecoville"]},
    )
    got = _extract_bairro(df)
    assert got.iloc[0] == "Mossunguê"


def test_extract_bairro_non_curitiba_returns_na():
    df = pd.DataFrame(
        {"Descricao": ["Casa em São Paulo"]},
    )
    got = _extract_bairro(df)
    assert pd.isna(got.iloc[0])


def test_extract_bairro_null_descricao():
    df = pd.DataFrame({"Descricao": [None]})
    got = _extract_bairro(df)
    assert pd.isna(got.iloc[0])


def test_extract_bairro_is_categorical():
    df = pd.DataFrame(
        {"Descricao": ["Apt Batel"]},
    )
    got = _extract_bairro(df)
    assert isinstance(got.dtype, pd.CategoricalDtype)


def test_extract_bairro_centro_civico_not_confused_with_centro():
    df = pd.DataFrame(
        {"Descricao": ["Lindo ap Centro Cívico"]},
    )
    got = _extract_bairro(df)
    assert got.iloc[0] == "Centro Cívico"


# ------------------------------------------------------------------ #
# _classify_category
# ------------------------------------------------------------------ #


def test_classify_category_apartamento_from_url():
    df = pd.DataFrame(
        {
            "URL": [
                "https://imovelweb.com.br/apartamentos/x",
            ],
            "Descricao": ["casa bonita"],
        },
    )
    got = _classify_category(df)
    assert got.iloc[0] == "Apartamento"


def test_classify_category_casa_from_url():
    df = pd.DataFrame(
        {
            "URL": [
                "https://imovelweb.com.br/casas/y",
            ],
            "Descricao": ["apartamento novo"],
        },
    )
    got = _classify_category(df)
    assert got.iloc[0] == "Casa"


def test_classify_category_fallback_to_descricao():
    df = pd.DataFrame(
        {
            "URL": [
                "https://imovelweb.com.br/propriedades/z",
            ],
            "Descricao": ["Lindo sobrado com 3 quartos"],
        },
    )
    got = _classify_category(df)
    assert got.iloc[0] == "Casa"


def test_classify_category_is_categorical():
    df = pd.DataFrame(
        {
            "URL": [
                "https://imovelweb.com.br/apartamentos/x",
            ],
            "Descricao": ["algo"],
        },
    )
    got = _classify_category(df)
    assert isinstance(got.dtype, pd.CategoricalDtype)


# ------------------------------------------------------------------ #
# _extract_amenities
# ------------------------------------------------------------------ #


def test_extract_amenities_detects_pool_and_gym():
    series = pd.Series(
        ["Condomínio com piscina e academia"],
    )
    got = _extract_amenities(series)
    assert got["Pool"].iloc[0] == 1
    assert got["Gym"].iloc[0] == 1


def test_extract_amenities_no_match():
    series = pd.Series(["Apartamento simples sem nada"])
    got = _extract_amenities(series)
    assert got.sum().sum() == 0


def test_extract_amenities_null_description():
    series = pd.Series([None])
    got = _extract_amenities(series)
    assert got.sum().sum() == 0


def test_extract_amenities_returns_correct_column_count():
    series = pd.Series(["Algo"])
    got = _extract_amenities(series)
    assert len(got.columns) == len(_AMENITY_PATTERNS)



# ------------------------------------------------------------------ #
# _detect_offplan
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "text",
    [
        "Apartamento na planta, entrega 2027",
        "Lançamento exclusivo no bairro",
        "Em construção, previsão de entrega",
        "Breve lançamento no centro",
    ],
)
def test_detect_offplan_keywords(text):
    series = pd.Series([text])
    got = _detect_offplan(series)
    assert got.iloc[0] == 1


def test_detect_offplan_not_offplan():
    series = pd.Series(["Apartamento pronto para morar"])
    got = _detect_offplan(series)
    assert got.iloc[0] == 0


def test_detect_offplan_null_returns_zero():
    series = pd.Series([None])
    got = _detect_offplan(series)
    assert got.iloc[0] == 0


# ------------------------------------------------------------------ #
# _create_count_dummies
# ------------------------------------------------------------------ #


def test_create_count_dummies_bedroom():
    df = pd.DataFrame({"N_quartos": [1, 2, 3, 4, 5]})
    result = _create_count_dummies(df, "N_quartos", "BED_", 4)
    assert result["BED_1"].tolist() == [0, 1, 0, 0, 0]
    assert result["BED_2"].tolist() == [0, 0, 1, 0, 0]
    assert result["BED_3"].tolist() == [0, 0, 0, 1, 0]
    assert result["BED_4"].tolist() == [0, 0, 0, 0, 1]


def test_create_count_dummies_parking():
    df = pd.DataFrame({"N_vagas": [0, 1, 2, 3, 4]})
    result = _create_count_dummies(df, "N_vagas", "PARK_", 2)
    assert result["PARK_1"].tolist() == [0, 0, 1, 0, 0]
    assert result["PARK_2"].tolist() == [0, 0, 0, 1, 1]


def test_create_count_dummies_null_treated_as_reference():
    df = pd.DataFrame({"col": [None, 3.0]})
    result = _create_count_dummies(df, "col", "D_", 2)
    assert result["D_1"].iloc[0] == 0
    assert result["D_2"].iloc[0] == 0


# ------------------------------------------------------------------ #
# _detect_outliers
# ------------------------------------------------------------------ #


def test_detect_outliers_extreme_price_flagged():
    df = pd.DataFrame(
        {
            "Preco": pd.array(
                [500_000, 600_000, 999_999_999],
                dtype=pd.Float32Dtype(),
            ),
            "Area_util_m2": pd.array(
                [80, 90, 100], dtype=pd.Float32Dtype(),
            ),
        },
    )
    got = _detect_outliers(df)
    assert got.iloc[2] == 1


def test_detect_outliers_normal_values_not_flagged():
    df = pd.DataFrame(
        {
            "Preco": pd.array(
                [400_000, 500_000, 600_000],
                dtype=pd.Float32Dtype(),
            ),
            "Area_util_m2": pd.array(
                [70, 80, 90], dtype=pd.Float32Dtype(),
            ),
        },
    )
    got = _detect_outliers(df)
    assert got.sum() == 0


# ------------------------------------------------------------------ #
# Validation
# ------------------------------------------------------------------ #


def test_fail_if_bairro_not_valid_raises():
    series = pd.Series(["Batel", "FakeBairro"])
    with pytest.raises(ValueError, match="Invalid bairros"):
        _fail_if_bairro_not_valid(series)


def test_fail_if_bairro_valid_passes():
    series = pd.Series(["Batel", "Centro"])
    _fail_if_bairro_not_valid(series)


def test_fail_if_price_negative_raises():
    series = pd.Series([-100.0, 500_000.0])
    with pytest.raises(ValueError, match="Negative prices"):
        _fail_if_price_negative(series)


def test_fail_if_price_positive_passes():
    series = pd.Series([100_000.0, 500_000.0])
    _fail_if_price_negative(series)


# ------------------------------------------------------------------ #
# Integration: full pipeline
# ------------------------------------------------------------------ #


def test_clean_housing_returns_dataframe(raw_df_minimal):
    result = clean_housing(raw_df_minimal)
    assert isinstance(result, pd.DataFrame)


def test_clean_housing_no_dropped_columns_remain(raw_df_minimal):
    result = clean_housing(raw_df_minimal)
    forbidden = {
        "URL", "Tipo", "Categoria", "Adicionais",
        "Areas_comuns", "Areas_privativas", "Descricao",
        "Planta", "IPTU", "Idade_anos", "Area_total_m2",
        "N_quartos", "N_banheiros", "N_vagas",
        "Endereco", "Bairro", "Preco",
    }
    assert forbidden.isdisjoint(set(result.columns))


def test_clean_housing_neighborhood_is_categorical(raw_df_minimal):
    result = clean_housing(raw_df_minimal)
    assert isinstance(
        result["neighborhood"].dtype, pd.CategoricalDtype,
    )


def test_clean_housing_price_is_float32(raw_df_minimal):
    result = clean_housing(raw_df_minimal)
    assert result["price"].dtype == pd.Float32Dtype()


def test_clean_housing_amenity_columns_present(raw_df_minimal):
    result = clean_housing(raw_df_minimal)
    expected_amenities = [
        "party_room", "game_room", "gym", "pool",
        "sauna", "bbq", "gourmet_space", "sports_court",
        "guardhouse", "cameras", "balcony", "playground",
    ]
    for col in expected_amenities:
        assert col in result.columns


def test_clean_housing_dummy_columns_present(raw_df_minimal):
    result = clean_housing(raw_df_minimal)
    expected = [
        "bedroom_1", "bedroom_2", "bedroom_3", "bedroom_4",
        "bathroom_1", "bathroom_2", "bathroom_3", "bathroom_4",
        "parking_1", "parking_2",
    ]
    for col in expected:
        assert col in result.columns


def test_clean_housing_outlier_column_present(raw_df_minimal):
    result = clean_housing(raw_df_minimal)
    assert "outlier" in result.columns


def test_clean_housing_no_null_neighborhood(raw_df_minimal):
    result = clean_housing(raw_df_minimal)
    assert result["neighborhood"].notna().all()
