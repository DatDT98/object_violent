import cv2
import numpy as np

from entities.common_entity import Vehicle, Box
from utils import error_code
from utils.customized_exception import BadRequestException
from utils.detect_vehicle import utils
from utils.ocr_plate.postprocess import postprocess
from utils.ocr_plate.preprocess import preprocess
from services.core_services.serving_service import ServingService
from utils.plate_detection.utils import im2single, decode_predict, reconstruct


class VehicleRecognitionImageService:
    def __init__(self, serving_service: ServingService):
        self.serving_service = serving_service
        self.license_plate_threshold = 0.5
        self.detect_vehicle_confidence_threshold = 0.8
        self.detect_vehicle_nms_threshold = 0.4
        self.classes = ['moto', 'truck', 'bicycle', 'car', 'person', 'van']

    def recognize_vehicle_and_license_plate(self, image: np.ndarray):
        vehicles = []
        bounding_boxes, confidences, label = self.detect_vehicle(image)
        for i, bbox in enumerate(bounding_boxes):
            x_min = bbox[0]
            y_min = bbox[1]
            width = bbox[2]
            height = bbox[3]
            x_max = bbox[0] + width
            y_max = bbox[1] + height
            vehicle_image = image[y_min: y_max, x_min: x_max]
            license_plate_text = self.extract_license_plate_from_vehicle_image(vehicle_image)
            recognized_vehicle = Vehicle(vehicle_type=label[i], license_plate=license_plate_text,
                                         bounding_box=Box(x_min, y_min, width, height))

            vehicles.append(recognized_vehicle)
        return vehicles

    def detect_one_license_plate_per_image(self, image_list):
        input_size = (366, 366)
        resized_images = []
        for image in image_list:
            resized_image = cv2.resize(image, input_size)
            normalized_image = im2single(resized_image)
            resized_images.append(normalized_image)
        resized_images = np.array(resized_images)
        response_matrix = self.serving_service.detect_license_plate(resized_images)

        # prepare response
        point_plate_list = []
        license_plate_images = []
        for i, image in enumerate(image_list):
            height, width, _ = image.shape
            final_labels = decode_predict(response_matrix[i], (input_size[0], input_size[1], 3),
                                          self.license_plate_threshold)
            if len(final_labels) > 0 and final_labels[0].prob() > 0.7:
                license_plate_image, point_label_plate = reconstruct(image, final_labels)
                point_plate = [int(point_label_plate.tl()[0] * width), int(point_label_plate.tl()[1] * height),
                               int(point_label_plate.br()[0] * width), int(point_label_plate.br()[1] * height),
                               final_labels[0].prob()]
            else:
                license_plate_image, point_plate = None, None

            point_plate_list.append(point_plate)
            license_plate_images.append(license_plate_image)
        return point_plate_list, license_plate_images

    def extract_text_from_license_plates(self, license_plate_images):
        input_images = []
        ocr_results = []
        response_probs = []
        for license_plate_image in license_plate_images:
            if license_plate_image is None:
                continue
            height, width, _ = license_plate_image.shape
            if width / height > 2.2:
                plate_shape = 1
            else:
                plate_shape = 2
            img = preprocess(license_plate_image, plate_shape)
            input_images.append(img)
        input_images = np.array(input_images)

        predictions, probs = self.serving_service.extract_text_from_license_plate(input_images)
        for i in range(0, len(license_plate_images)):
            result = ''
            for character in predictions[i].decode("utf-8"):
                if not character.isalpha() and not character.isdigit() and character != '#':
                    break
                result += character
            text, min_prob = postprocess(result, probs[i].max(1))
            text = text.replace('#', 'D')
            ocr_results.append(text)
            response_probs.append(min_prob)
        return ocr_results, response_probs

    def detect_vehicle(self, image_matrix):
        image_height, image_width, _ = image_matrix.shape
        resized_image = cv2.cvtColor(cv2.resize(image_matrix, (416, 416)), cv2.COLOR_BGR2RGB)
        if type(resized_image) == np.ndarray and len(resized_image.shape) == 3:  # cv2 image
            input_image = resized_image / 255.0
            input_image = np.transpose(input_image, (2, 0, 1))
            input_image = np.expand_dims(input_image, axis=0)
        elif type(resized_image) == np.ndarray and len(resized_image.shape) == 4:
            input_image = np.transpose(resized_image, (0, 3, 1, 2)) / 255.0
        else:
            raise BadRequestException(error_code.INVALID_IMAGE_BASE64, "image_base64")

        output = self.serving_service.detect_vehicle(input_image)
        bounding_boxes = utils.post_processing(self.detect_vehicle_confidence_threshold,
                                               self.detect_vehicle_nms_threshold, output)
        response_bounding_boxes = []
        confidences = []
        labels = []
        for box in bounding_boxes[0]:
            if box[5] > 0.6:
                class_id = box[6]
                x_min = int(box[0] * image_width)
                if x_min < 0:
                    x_min = 0
                y_min = int(box[1] * image_height)
                if y_min < 0:
                    y_min = 0
                x_max = int(box[2] * image_width)
                y_max = int(box[3] * image_height)
                width = x_max - x_min
                height = y_max - y_min
                response_bounding_boxes.append([x_min, y_min, width, height])
                confidences.append(box[5])
                labels.append(self.classes[int(class_id)])
        return response_bounding_boxes, confidences, labels

    def extract_license_plate_from_vehicle_image(self, vehicle_image):
        point_plates, image_plates = self.detect_one_license_plate_per_image([vehicle_image])
        if image_plates[0] is None:
            return ''
        texts, _ = self.extract_text_from_license_plates(image_plates)
        return texts[0]
