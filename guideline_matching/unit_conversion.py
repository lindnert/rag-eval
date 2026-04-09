import pint
from enum import Enum
from model import Timeframe, TIMEFRAME_TO_DAYS, Nutrient

ureg = pint.UnitRegistry()

class NutrientUnit(str, Enum):
    G = "g"
    MG = "mg"
    UG = "µg"          # microgram
    KCAL = "kcal"
    KJ = "kJ"
    ML = "ml"

CUSTOM_UNITS = {
    "%E": "percent_of_energy",
    "%DV": "percent_daily_value",
    "portions": "portions",
    "IU": "international_units",
}

NUTRIENT_ALIASES: dict[str, Nutrient] = {
    # English aliases
    "energy": Nutrient.ENERGY,
    "calories": Nutrient.ENERGY,
    "protein": Nutrient.PROTEIN,
    "total fat": Nutrient.FAT_TOTAL,
    "fat": Nutrient.FAT_TOTAL,
    "saturated fat": Nutrient.FAT_SATURATED,
    "monounsaturated fat": Nutrient.FAT_UNSATURATED_MONO,
    "polyunsaturated fat": Nutrient.FAT_UNSATURATED_POLY,
    "trans fat": Nutrient.FAT_TRANS,
    "carbohydrates": Nutrient.CARBOHYDRATES,
    "carbs": Nutrient.CARBOHYDRATES,
    "sugar": Nutrient.SUGAR,
    "fiber": Nutrient.FIBER,
    "fibre": Nutrient.FIBER,
    "dietary fiber": Nutrient.FIBER,
    "water": Nutrient.WATER,
    "calcium": Nutrient.CALCIUM,
    "iron": Nutrient.IRON,
    "magnesium": Nutrient.MAGNESIUM,
    "phosphorus": Nutrient.PHOSPHORUS,
    "potassium": Nutrient.POTASSIUM,
    "sodium": Nutrient.SODIUM,
    "zinc": Nutrient.ZINC,
    "selenium": Nutrient.SELENIUM,
    "iodine": Nutrient.IODINE,
    "copper": Nutrient.COPPER,
    "manganese": Nutrient.MANGANESE,
    "chromium": Nutrient.CHROMIUM,
    "molybdenum": Nutrient.MOLYBDENUM,
    "fluoride": Nutrient.FLUORIDE,
    "chloride": Nutrient.CHLORIDE,
    "folate": Nutrient.VITAMIN_B9,
    "folic acid": Nutrient.VITAMIN_B9,
    "thiamine": Nutrient.VITAMIN_B1,
    "riboflavin": Nutrient.VITAMIN_B2,
    "niacin": Nutrient.VITAMIN_B3,
    "pantothenic acid": Nutrient.VITAMIN_B5,
    "biotin": Nutrient.VITAMIN_B7,
    "cobalamin": Nutrient.VITAMIN_B12,
    "retinol": Nutrient.VITAMIN_A,
    "ascorbic acid": Nutrient.VITAMIN_C,
    "cholecalciferol": Nutrient.VITAMIN_D,
    "tocopherol": Nutrient.VITAMIN_E,
    "omega-3": Nutrient.OMEGA_3,
    "omega-3 fatty acids": Nutrient.OMEGA_3,
    "omega-6": Nutrient.OMEGA_6,
    "omega-6 fatty acids": Nutrient.OMEGA_6,
    "DHA": Nutrient.DHA,
    "EPA": Nutrient.EPA,
    "cholesterol": Nutrient.CHOLESTEROL,
    "alcohol": Nutrient.ALCOHOL,
    "salt": Nutrient.SALT,
    "table salt": Nutrient.SALT,
    "NaCl": Nutrient.SALT,
    "Na": Nutrient.SODIUM,
    "Fe": Nutrient.IRON,
    "Ca": Nutrient.CALCIUM,
    "Zn": Nutrient.ZINC,
    "Se": Nutrient.SELENIUM,

    # German aliases
    "Energie": Nutrient.ENERGY,
    "Kalorien": Nutrient.ENERGY,
    "Brennwert": Nutrient.ENERGY,
    "Eiweiß": Nutrient.PROTEIN,
    "Protein": Nutrient.PROTEIN,
    "Gesamtfett": Nutrient.FAT_TOTAL,
    "Fett": Nutrient.FAT_TOTAL,
    "gesättigte Fettsäuren": Nutrient.FAT_SATURATED,
    "einfach ungesättigte Fettsäuren": Nutrient.FAT_UNSATURATED_MONO,
    "mehrfach ungesättigte Fettsäuren": Nutrient.FAT_UNSATURATED_POLY,
    "Transfettsäuren": Nutrient.FAT_TRANS,
    "Transfette": Nutrient.FAT_TRANS,
    "Kohlenhydrate": Nutrient.CARBOHYDRATES,
    "Zucker": Nutrient.SUGAR,
    "Ballaststoffe": Nutrient.FIBER,
    "Wasser": Nutrient.WATER,
    "Kalzium": Nutrient.CALCIUM,
    "Calcium": Nutrient.CALCIUM,
    "Eisen": Nutrient.IRON,
    "Magnesium": Nutrient.MAGNESIUM,
    "Phosphor": Nutrient.PHOSPHORUS,
    "Kalium": Nutrient.POTASSIUM,
    "Natrium": Nutrient.SODIUM,
    "Zink": Nutrient.ZINC,
    "Selen": Nutrient.SELENIUM,
    "Jod": Nutrient.IODINE,
    "Iod": Nutrient.IODINE,
    "Kupfer": Nutrient.COPPER,
    "Mangan": Nutrient.MANGANESE,
    "Chrom": Nutrient.CHROMIUM,
    "Molybdän": Nutrient.MOLYBDENUM,
    "Fluorid": Nutrient.FLUORIDE,
    "Chlorid": Nutrient.CHLORIDE,
    "Folat": Nutrient.VITAMIN_B9,
    "Folsäure": Nutrient.VITAMIN_B9,
    "Thiamin": Nutrient.VITAMIN_B1,
    "Riboflavin": Nutrient.VITAMIN_B2,
    "Niacin": Nutrient.VITAMIN_B3,
    "Pantothensäure": Nutrient.VITAMIN_B5,
    "Biotin": Nutrient.VITAMIN_B7,
    "Cobalamin": Nutrient.VITAMIN_B12,
    "Retinol": Nutrient.VITAMIN_A,
    "Ascorbinsäure": Nutrient.VITAMIN_C,
    "Cholecalciferol": Nutrient.VITAMIN_D,
    "Tocopherol": Nutrient.VITAMIN_E,
    "Vitamin A": Nutrient.VITAMIN_A,
    "Vitamin D": Nutrient.VITAMIN_D,
    "Vitamin E": Nutrient.VITAMIN_E,
    "Vitamin K": Nutrient.VITAMIN_K,
    "Vitamin C": Nutrient.VITAMIN_C,
    "Vitamin B1": Nutrient.VITAMIN_B1,
    "Vitamin B2": Nutrient.VITAMIN_B2,
    "Vitamin B3": Nutrient.VITAMIN_B3,
    "Vitamin B5": Nutrient.VITAMIN_B5,
    "Vitamin B6": Nutrient.VITAMIN_B6,
    "Vitamin B7": Nutrient.VITAMIN_B7,
    "Vitamin B9": Nutrient.VITAMIN_B9,
    "Vitamin B12": Nutrient.VITAMIN_B12,
    "Omega-3-Fettsäuren": Nutrient.OMEGA_3,
    "Omega-6-Fettsäuren": Nutrient.OMEGA_6,
    "Cholesterin": Nutrient.CHOLESTEROL,
    "Alkohol": Nutrient.ALCOHOL,
    "Salz": Nutrient.SALT,
    "Speisesalz": Nutrient.SALT,
    "Kochsalz": Nutrient.SALT,
}

def normalize_amount(amount: float, unit: str, target_unit: str) -> float:
    """Convert using Pint where possible, fall back to custom units."""
    if unit in CUSTOM_UNITS or target_unit in CUSTOM_UNITS:
        if unit == target_unit:
            return amount
        raise ValueError(f"Cannot convert {unit} to {target_unit}")
    qty = amount * ureg(unit)
    return qty.to(target_unit).magnitude

# Example usage:
qty = ureg.Quantity(500, "mg")
converted = qty.to("g")  # → 0.5 g

def normalize_to_daily(amount: float, timeframe: Timeframe) -> float:
    if timeframe in (Timeframe.PER_MEAL, Timeframe.PER_SERVING):
        raise ValueError(f"Cannot convert {timeframe} to daily without meal count")
    return amount / TIMEFRAME_TO_DAYS[timeframe]