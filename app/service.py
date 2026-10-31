import pandas as pd

from churnguard import config, explain, registry


class PredictionService:
    def __init__(self, pipeline, meta):
        self.pipeline = pipeline
        self.meta = meta
        self.threshold = float(meta["threshold"])
        self.version = str(meta["tag"])

    @classmethod
    def load(cls):
        pipeline, meta = registry.load_or_train()
        return cls(pipeline, meta)

    def label_for(self, probability):
        return "Yes" if probability >= self.threshold else "No"

    def risk_level(self, probability):
        if probability >= max(0.5, self.threshold):
            return "high"
        if probability >= self.threshold:
            return "medium"
        return "low"

    def to_frame(self, records):
        return pd.DataFrame(records)[config.RAW_FEATURES]

    def predict_one(self, record, top_k=3):
        frame = self.to_frame([record])
        probability = float(self.pipeline.predict_proba(frame)[0, 1])
        drivers = explain.explain(self.pipeline, frame, top_k=top_k)[0]
        return {
            "probability": round(probability, 4),
            "label": self.label_for(probability),
            "risk_level": self.risk_level(probability),
            "threshold": round(self.threshold, 4),
            "top_drivers": drivers,
            "summary": " ".join(driver["explanation"] for driver in drivers),
            "model_version": self.version,
        }

    def predict_many(self, records):
        frame = self.to_frame(records)
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        return [
            {
                "index": index,
                "probability": round(float(probability), 4),
                "label": self.label_for(float(probability)),
                "risk_level": self.risk_level(float(probability)),
            }
            for index, probability in enumerate(probabilities)
        ]
