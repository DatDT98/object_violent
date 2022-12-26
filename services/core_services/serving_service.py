from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
import grpc

from utils.application_properties import get_config_variable


class ServingService:
    def __init__(self):
        self.license_plate_host = get_config_variable("tensorflow_serving_host")
        self.stub = self.get_license_plate_stub()

    def get_license_plate_stub(self):
        channel = grpc.insecure_channel(self.license_plate_host)
        return prediction_service_pb2_grpc.PredictionServiceStub(channel)

    def detect_license_plate(self, image_matrix):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'detect_plate'
        request.model_spec.signature_name = 'serving_default'

        request.inputs['image'].CopyFrom(tf.make_tensor_proto(image_matrix, dtype=tf.float32))
        request.inputs['is_training'].CopyFrom(tf.make_tensor_proto(False, dtype=tf.bool))
        response = self.stub.Predict(request, 10.0)  # 10 seconds
        outputs = tf.make_ndarray(response.outputs['concatenate_1/concat:0'])
        return outputs

    def extract_text_from_license_plate(self, image_matrix):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'ocr_plate'
        request.model_spec.signature_name = 'serving_default'

        request.inputs['image'].CopyFrom(tf.make_tensor_proto(image_matrix, dtype=tf.uint8))
        response = self.stub.Predict(request, 10.0)  # 10 seconds
        predictions = response.outputs['predicted_text'].string_val
        prob = tf.make_ndarray(response.outputs['prob'])
        return predictions, prob

    def detect_vehicle(self, image_matrix):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'detect_vehicle'
        request.model_spec.signature_name = 'serving_default'

        request.inputs['input'].CopyFrom(tf.make_tensor_proto(image_matrix, dtype=tf.float32))
        response = self.stub.Predict(request, 10.0)  # 10 seconds
        output1 = tf.make_ndarray(response.outputs['output1'])
        output2 = tf.make_ndarray(response.outputs['output2'])
        return [output1, output2]

    def extract_tracking_features(self, image_matrix):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'tracking'
        request.model_spec.signature_name = 'serving_default'

        request.inputs['images'].CopyFrom(tf.make_tensor_proto(image_matrix, dtype=tf.uint8))
        response = self.stub.Predict(request, 10.0)  # 10 seconds
        features = tf.make_ndarray(response.outputs['features'])
        return features
