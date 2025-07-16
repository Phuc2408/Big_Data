import json
import time
from kafka import KafkaProducer

# --- Cấu hình ---
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "text-comments"
INPUT_FILE = "E:\\Big_Data\\input_for_kafka.json" 

# Tạo một Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print(f"Bắt đầu gửi dữ liệu từ '{INPUT_FILE}' đến topic '{KAFKA_TOPIC}'...")

# Đọc toàn bộ nội dung file và parse JSON một lần
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        json_content = f.read()
        comments_data = json.loads(json_content)
    
    # Lặp qua từng đối tượng trong list và gửi lên Kafka
    for comment_data in comments_data:
        print(f"Đang gửi: {comment_data}")
        producer.send(KAFKA_TOPIC, value=comment_data)
        time.sleep(2) 

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{INPUT_FILE}'")
except json.JSONDecodeError as e:
    print(f"Lỗi khi parse JSON từ file '{INPUT_FILE}': {e}")
    print("Vui lòng đảm bảo file là một mảng JSON hợp lệ.")
except Exception as e:
    print(f"Gặp lỗi không mong muốn: {e}")

# Đảm bảo tất cả message đã được gửi đi trước khi thoát
producer.flush()
print("Đã gửi xong tất cả dữ liệu.")