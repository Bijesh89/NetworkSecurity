from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.components.model_trainer import ModelTrainer
import sys

if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifect = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed and artifact is created")
        print(dataingestionartifect)
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(
                                        data_validation_config=data_validation_config,
                                        data_ingestion_artifact=dataingestionartifect 
                                    )
        logging.info("Initiate the data validation")
        data_validation_artifect = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifect)
        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        logging.info("Initiate the data transformation")
        data_transformation = DataTransformation(
                                            data_validation_artifact=data_validation_artifect,
                                            data_transformation_config=data_transformation_config
                                        )
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed")
        print(data_transformation_artifact)

        logging.info("Initiate the model trainer component")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config=trainingpipelineconfig)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model trainer component completed")
        

    except Exception as e:
        raise NetworkSecurityException(e, sys)
