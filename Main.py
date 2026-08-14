import DataLoading
import EDA
import Cleaning
import DataTypes
import FeatureEngineering
import Correlation
import Encoding
import Scaling
import SMOTEBalancing
import TrainTestSplit
import DataPreparation
import TreeModels
import ClassicalModels
import DeepLearningModels
import Evaluation
import ROC
import FeatureImportance
import FinalEnsemble


def main():
    DataLoading.main()
    EDA.main()
    Cleaning.main()
    DataTypes.main()
    FeatureEngineering.main()
    Correlation.main()
    Encoding.main()
    Scaling.main()
    SMOTEBalancing.main()
    TrainTestSplit.main()
    DataPreparation.main()
    TreeModels.main()
    ClassicalModels.main()
    DeepLearningModels.main()
    Evaluation.main()
    ROC.main()
    FeatureImportance.main()
    FinalEnsemble.main()
    
if __name__ == "__main__":
    main()