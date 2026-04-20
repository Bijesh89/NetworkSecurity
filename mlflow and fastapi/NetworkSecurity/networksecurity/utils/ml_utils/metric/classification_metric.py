from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from sklearn.metrics import f1_score,precision_score,recall_score
from networksecurity.exception.exception import NetworkSecurityException
import sys

def get_classification_score(y_true,y_pred) -> ClassificationMetricArtifact:
    try:
        model_f1_score=f1_score(y_true,y_pred)
        model_precision=precision_score(y_true,y_pred)
        model_recall=recall_score(y_true,y_pred)

        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision,
            recall_score=model_recall
        )

        return classification_metric

    except Exception as e:
        raise NetworkSecurityException(e,sys)