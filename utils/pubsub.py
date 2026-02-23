import os

from dotenv import load_dotenv
from google.cloud import pubsub_v1

load_dotenv()

publisher = pubsub_v1.PublisherClient()

def publish_to_topic (topic_id: str,
                      message: str):

    topic_name = 'projects/{project_id}/topics/{topic}'.format(
        project_id=os.getenv('GC_PROJECT_ID'),
        topic=topic_id,
    )

    data = message.encode("utf-8")
    result = publisher.publish(topic_name, data)

    result.result()

    print(f"Message {result} has been published.")
