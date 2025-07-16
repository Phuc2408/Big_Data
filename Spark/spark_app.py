import torch
from torch import nn
import json
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, pandas_udf, to_json, struct, udf
from pyspark.sql.types import StructType, StructField, StringType, BinaryType 
from transformers import AutoTokenizer, AutoModel

# Import các lớp cần thiết cho mô hình PyTorch Lightning
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import lightning.pytorch as pl
from torchmetrics.classification import MulticlassF1Score

# --- Định nghĩa lớp mô hình PyTorch Lightning ---
class XLMRobertaCommentClassifier(pl.LightningModule):
    def __init__(self, num_classes: int, num_predictions: int = 5, lr: float = 2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.lm = AutoModel.from_pretrained("uitnlp/CafeBERT")
        self.cls = nn.Linear(1024, self.hparams.num_classes * self.hparams.num_predictions)
        self.criterion = CrossEntropyLoss(ignore_index=-1)
        self.val_f1 = MulticlassF1Score(num_classes=self.hparams.num_classes, average='micro', ignore_index=-1)
        self.test_f1 = MulticlassF1Score(num_classes=self.hparams.num_classes, average='micro', ignore_index=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.lm(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"][:, 0, :]
        x = self.cls(x)
        return x

    def _process_batch(self, batch):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        reshaped_logits = logits.view(-1, self.hparams.num_classes)
        reshaped_labels = batch["labels"].view(-1)
        loss = self.criterion(reshaped_logits, reshaped_labels)
        return loss, reshaped_logits, reshaped_labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._process_batch(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._process_batch(batch)
        self.val_f1(logits, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._process_batch(batch)
        self.test_f1(logits, labels)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)

# === Đường dẫn file ===
DRIVE_PROJECT_PATH = 'E:\\Big_Data'
MODEL_WEIGHTS_PATH = f"{DRIVE_PROJECT_PATH}\\model_weights.pt"
LABELMAP_PATH = f"{DRIVE_PROJECT_PATH}\\labelmap.json"

# --- Pandas UDF ---
_tokenizer = None
_model = None
_id_to_label = None
_num_classes = None
_device = None

def create_prediction_udf():
    global _tokenizer, _model, _id_to_label, _num_classes, _device

    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing model on worker, using device: {_device}")

        try:
            with open(LABELMAP_PATH, "r", encoding="utf-8") as f:
                _id_to_label = {int(k): v for k, v in json.load(f).items()}
                _num_classes = len(_id_to_label)
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file labelmap.json tại {LABELMAP_PATH}")
            raise

        _tokenizer = AutoTokenizer.from_pretrained("uitnlp/CafeBERT", use_fast=False)
        print("Tokenizer created.")

        _model = XLMRobertaCommentClassifier(num_classes=_num_classes, num_predictions=5)
        try:
            _model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=_device))
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file model_weights.pt tại {MODEL_WEIGHTS_PATH}")
            raise
        except Exception as e:
            print(f"Lỗi khi tải model state_dict: {e}")
            raise

        _model.to(_device)
        _model.eval()
        print("Model initialized successfully on worker.")
    else:
        print("Model and tokenizer already initialized on worker.")

    @pandas_udf(StringType())
    def predict_udf(texts: pd.Series) -> pd.Series:
        if texts.empty:
            return pd.Series([], dtype=str)

        text_list = [text for text in texts if pd.notna(text)]
        if not text_list:
            return pd.Series([""] * len(texts), dtype=str)

        encoded = _tokenizer(
            text_list,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(_device)
        attention_mask = encoded["attention_mask"].to(_device)

        with torch.no_grad():
            logits = _model(input_ids, attention_mask)
            logits = logits.view(-1, _num_classes)
            preds = torch.argmax(logits, dim=1).view(-1, _model.hparams.num_predictions).cpu()

        results = []
        for row_preds in preds:
            labels = [_id_to_label.get(p.item(), "unknown") for p in row_preds]
            filtered_labels = [label for label in labels if label != "product#null#neutral"]
            results.append(", ".join(filtered_labels))
        
        final_results = []
        result_idx = 0
        for text_original in texts:
            if pd.notna(text_original):
                final_results.append(results[result_idx])
                result_idx += 1
            else:
                final_results.append("")
        
        return pd.Series(final_results, dtype=str)

    return predict_udf

# --- Hàm Main ---
def main():
    spark = SparkSession.builder \
        .appName("Ứng dụng Phân loại văn bản Spark") \
        .master("local[*]") \
        .config("spark.driver.memory", "16g") \
        .config("spark.executor.memory", "16g") \
        .config("spark.driver.maxResultSize", "0") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -Dfile.encoding=UTF-8") \
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -Dfile.encoding=UTF-8") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.4") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print("SparkSession configured with increased memory.")
    print("Starting Spark app with Pandas UDF...")

    classify_udf = create_prediction_udf()
    text_schema = StructType([StructField("text", StringType(), True)])

    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "text-comments") \
        .option("startingOffsets", "earliest") \
        .load()

    def decode_binary_to_string(binary_data):
        if binary_data is not None:
            return binary_data.decode('utf-8', errors='replace')
        return None

    decode_udf = udf(decode_binary_to_string, StringType())

    decoded_kafka_df = kafka_df.select(
        decode_udf(col("value")).alias("decoded_value_string")
    )

    parsed_df = decoded_kafka_df.select(
        from_json(col("decoded_value_string"), text_schema).alias("data")
    ).select("data.text")


    result_df = parsed_df.withColumn("prediction", classify_udf(col("text")))

    json_df = result_df.select(
        to_json(struct(col("text"), col("prediction"))).alias("value")
    )

    query = result_df.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", "false")\
        .start()
        

    print("\nStreaming query started. Output will be in JSON format.")
    print("Waiting for messages from Kafka...")
    query.awaitTermination()
    print("\nStreaming query terminated.")
    return query

if __name__ == "__main__":
    import os
    import findspark
    
    findspark.init()
    print(f"SPARK_HOME is set to: {os.environ.get('SPARK_HOME')}")
    print(f"JAVA_HOME is set to: {os.environ.get('JAVA_HOME')}")
    print("Findspark initialized.")

    streaming_query = main()