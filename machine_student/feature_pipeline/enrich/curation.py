from typing import Dict, List, Any, Optional, Tuple

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from .encoders import Encoder
from .imputers import Imputer

class EnrichmentProcesss:
    def __init__(self,
                 columns: List[str],
                 imputation: Optional[str], 
                 imputation_kwargs: Optional[Dict[str, Any]],
                 encoding: Optional[str],
                 encoding_kwargs: Optional[Dict[str, Any]]
                 ) -> None:
        self.columns = columns
        self.imputation = imputation
        self.imputation_kwargs = imputation_kwargs
        self.encoding = encoding
        self.encoding_kwargs = encoding_kwargs
    
    def get_encoder(self) -> TransformerMixin:
        if self.encoding:
            encoder_class = Encoder(self.encoding, self.encoding_kwargs)
            encoder_estimator = encoder_class.get_estimator()
            return encoder_estimator
        else:
            return self.encoding

    def get_imputer(self) -> TransformerMixin:
        if self.imputation:
            imputer_class = Imputer(self.imputation, self.imputation_kwargs)
            imputer_estimator = imputer_class.get_estimator()
            return imputer_estimator
        else:
            return self.imputation

    def get_pipeline(self) -> Pipeline:
        imputer, encoder = self.get_imputer(), self.get_encoder()
        steps = [imputer, encoder]
        final_steps = [step for step in steps if step]
        pipeline = Pipeline(steps=final_steps)
        return pipeline
    
    def get_transformation_name(self) -> str:
        imp_part = self.imputation if self.imputation else "none"
        enc_part = self.encoding if self.encoding else "none"
        return f"{imp_part}_{enc_part}"

    def get_transformation(self) -> Tuple[str, Pipeline, List[str]]:
        name = self.get_transformation_name()
        transformation = (name, self.get_pipeline(), self.columns)
        return transformation
    
    
class EnrichmentPipeline:
    CONFIG = {
        "verbose": True,
        "remainder": "passthrough"
    }
    def __init__(self, pipeline_config: List[Dict[str,Any]]) -> None:
        self.pipeline_config = pipeline_config
 
    def get_transformations(self):
        processes = [EnrichmentProcesss(**config) for config in self.pipeline_config]
        transformers = [process.get_transformation() for process in processes]
        return transformers
        
    def get(self):
        transformers = self.get_transformations()
        pipeline = ColumnTransformer(transformers=transformers, remainder="passthrough")
        
