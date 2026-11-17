"""Reproduce the public fixed-rate green MTN pilot from the versioned inputs."""
import csv
from pathlib import Path
from statistics import mean, median

from rules import underwriter_price

CASE_FILE = Path(__file__).resolve().with_name("public_mtn_cases.csv")


def government_yield(case):
    years = int(case["term"][:-1])
    two_year = float(case["china_gov_2y"])
    five_year = float(case["china_gov_5y"])
    return two_year + (years - 2) * (five_year - two_year) / 3


def evaluate(calibration, validation):
    constant = median(float(case["coupon"]) for case in calibration)
    spread = median(float(case["coupon"]) - government_yield(case) for case in calibration)
    legacy = underwriter_price({"credit_rating": "AAA", "green_level": "high"}, "moderate")[2]["final_rate"]
    absolute_errors = [
        (
            abs(float(case["coupon"]) - legacy),
            abs(float(case["coupon"]) - constant),
            abs(float(case["coupon"]) - government_yield(case) - spread),
        )
        for case in validation
    ]
    print(f"Calibration {len(calibration)} | Validation {len(validation)}")
    print(f"Median coupon {constant:.3%} | Median spread {spread * 10000:.2f} bp")
    for label, errors in zip(("Existing rule", "Median coupon", "Government yield plus spread"), zip(*absolute_errors)):
        print(f"{label}: MAE {mean(errors) * 10000:.2f} bp")


if __name__ == "__main__":
    with CASE_FILE.open(encoding="utf-8", newline="") as file:
        cases = list(csv.DictReader(file))
    calibration = [case for case in cases if case["sample_role"] == "校准"]
    validation = [case for case in cases if case["sample_role"] == "验证"]
    evaluate(calibration, validation)
    validation_issuers = {case["issuer"] for case in validation}
    print("Calibration excluding issuers present in validation")
    evaluate([case for case in calibration if case["issuer"] not in validation_issuers], validation)
