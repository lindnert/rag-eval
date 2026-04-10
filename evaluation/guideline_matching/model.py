from pydantic import BaseModel
from enum import Enum
from typing import Literal, Optional

class Nutrient(str, Enum):
    # Macronutrients
    ENERGY = "energy"
    PROTEIN = "protein"
    FAT_TOTAL = "fat_total"
    FAT_SATURATED = "fat_saturated"
    FAT_UNSATURATED_MONO = "fat_unsaturated_mono"
    FAT_UNSATURATED_POLY = "fat_unsaturated_poly"
    FAT_TRANS = "fat_trans"
    CARBOHYDRATES = "carbohydrates"
    SUGAR = "sugar"
    FIBER = "fiber"
    WATER = "water"

    # Minerals
    CALCIUM = "calcium"
    IRON = "iron"
    MAGNESIUM = "magnesium"
    PHOSPHORUS = "phosphorus"
    POTASSIUM = "potassium"
    SODIUM = "sodium"
    ZINC = "zinc"
    SELENIUM = "selenium"
    IODINE = "iodine"
    COPPER = "copper"
    MANGANESE = "manganese"
    CHROMIUM = "chromium"
    MOLYBDENUM = "molybdenum"
    FLUORIDE = "fluoride"
    CHLORIDE = "chloride"

    # Fat-soluble vitamins
    VITAMIN_A = "vitamin_a"
    VITAMIN_D = "vitamin_d"
    VITAMIN_E = "vitamin_e"
    VITAMIN_K = "vitamin_k"

    # Water-soluble vitamins
    VITAMIN_C = "vitamin_c"
    VITAMIN_B1 = "vitamin_b1"    # thiamine
    VITAMIN_B2 = "vitamin_b2"    # riboflavin
    VITAMIN_B3 = "vitamin_b3"    # niacin
    VITAMIN_B5 = "vitamin_b5"    # pantothenic acid
    VITAMIN_B6 = "vitamin_b6"
    VITAMIN_B7 = "vitamin_b7"    # biotin
    VITAMIN_B9 = "vitamin_b9"    # folate
    VITAMIN_B12 = "vitamin_b12"  # cobalamin

    # Fatty acids (if your guidelines get this specific)
    OMEGA_3 = "omega_3"
    OMEGA_6 = "omega_6"
    DHA = "dha"
    EPA = "epa"

    # Other
    CHOLESTEROL = "cholesterol"
    ALCOHOL = "alcohol"
    SALT = "salt"               # sometimes listed separately from sodium

class RecommendationType(str, Enum):
    TARGET = "target"           # a single recommended value (= recommended_intake)
    RANGE = "range"             # lower and/or upper bound
    ADEQUATE_INTAKE = "adequate_intake"  # sufficient but not necessarily optimal

class PopulationGroup(BaseModel):
    sex: Optional[Literal["male", "female"]] = None
    age_min: Optional[int] = None  # inclusive
    age_max: Optional[int] = None  # inclusive
    life_stage: Optional[Literal[
        "infant", "child", "adolescent", "adult",
        "pregnant", "lactating", "postmenopausal"
    ]] = None

class Timeframe(str, Enum):
    PER_DAY = "per_day"
    PER_WEEK = "per_week"
    PER_MONTH = "per_month"
    PER_MEAL = "per_meal"
    PER_SERVING = "per_serving"

TIMEFRAME_TO_DAYS = {
    Timeframe.PER_DAY: 1,
    Timeframe.PER_WEEK: 7,
    Timeframe.PER_MONTH: 30,
}

class NutrientRecommendation(BaseModel):
    nutrient: str                          # canonical ID, e.g. "vitamin_d"
    amount: float
    amount_upper: Optional[float] = None   # for ranges
    unit: str                              # "mg", "µg", "g", "%E", "kcal"
    recommendation_type: RecommendationType
    population_group: Optional[PopulationGroup] = None # "adult_female_50-65"
    condition: Optional[str] = None        # "breast_cancer"
    timeframe: Optional[Timeframe] = Timeframe.PER_DAY
    source: str                            # "DGE_2024" or "rag_output"
    confidence: Optional[float] = None     # extraction confidence, RAG side only

class Compatibility(str, Enum):
    FULL = "full"
    PARTIAL = "partial"

COMPATIBILITY_MATRIX: dict[tuple[RecommendationType, RecommendationType], Compatibility] = {
    (RecommendationType.TARGET, RecommendationType.TARGET): Compatibility.FULL,
    (RecommendationType.TARGET, RecommendationType.ADEQUATE_INTAKE): Compatibility.FULL,
    (RecommendationType.TARGET, RecommendationType.RANGE): Compatibility.PARTIAL,
    (RecommendationType.RANGE, RecommendationType.RANGE): Compatibility.FULL,
    (RecommendationType.RANGE, RecommendationType.ADEQUATE_INTAKE): Compatibility.PARTIAL,
    (RecommendationType.ADEQUATE_INTAKE, RecommendationType.ADEQUATE_INTAKE): Compatibility.FULL,
}