import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

for _d in (DATA_DIR, ARTIFACT_DIR, MODEL_DIR, FIGURE_DIR):
    os.makedirs(_d, exist_ok=True)

TRAIN_RAW_PATH = os.path.join(DATA_DIR, "train_u6lujuX_CVtuZ9i.csv")
TEST_RAW_PATH = os.path.join(DATA_DIR, "test_Y3wMUE5_7gLdaTN.csv")
SEED_SPLIT_1 = 12345
SEED_SPLIT_2 = 42
SEED_MODELS_1 = 16
SEED_MODELS_2 = 16
SEED_DL = 14

CAT_VARS = ["Gender", "Married", "Dependents", "Self_Employed",
            "Credit_History", "Property_Area"]

NUM_VARS = ["LoanAmount", "Loan_Amount_Term", "ApplicantIncome", "CoapplicantIncome"]

ORIGINAL_COLS = ["Loan_ID", "Gender", "Married", "Dependents", "Education",
                  "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
                  "LoanAmount", "Loan_Amount_Term", "Credit_History",
                  "Property_Area", "Loan_Status", "TotalIncome"]

INTEGER_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                 "Loan_Amount_Term", "Credit_History"]

CHARACTER_COLS = ["Gender", "Married", "Education", "Self_Employed",
                   "Property_Area", "Loan_Status", "Dependents", "Loan_ID"]

CAT_COLS_TO_ENCODE = ["Gender", "Married", "Dependents", "Education",
                        "Self_Employed", "Property_Area", "Credit_History"]

NUM_COLS_FOR_SCALING = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "ApplicantIncome_Log", "CoapplicantIncome_Log", "LoanAmount_Log",
    "Loan_Amount_Term_Log", "TotalIncome", "TotalIncome_Log",
]

CORR_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
             "Credit_History", "TotalIncome", "ApplicantIncome_Log", "CoapplicantIncome_Log",
             "LoanAmount_Log", "Loan_Amount_Term_Log", "TotalIncome_Log"]

MODEL_FEATURE_COLS = ["Gender", "Married", "Dependents", "Education",
                        "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
                        "LoanAmount", "Loan_Amount_Term", "Credit_History",
                        "Property_Area"]

TARGET_COL = "Loan_Status"