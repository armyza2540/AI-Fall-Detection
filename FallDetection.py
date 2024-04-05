# Imports required libraries for the system
import sys

stdout = sys.stdout
import torch
from super_gradients.training import Trainer  # in charge of training, evaluation, saving checkpoints, etc
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, \
    coco_detection_yolo_format_val
from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training import MultiGPUMode
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.utils.distributed_training_utils import setup_device
from IPython.display import clear_output
import cv2
import os

sys.stdout = stdout


class FallDetector():
    def __init__(self, trained_model_path=None, model_name='yolo_nas_m', pretrained_weights='coco'):
        self.CHECKPOINT_DIR = "model_checkpoints"
        self.EXPERIMENT_NAME = 'fall_detection_experiment_v3'
        self.TRAINED_MODEL_PATH = trained_model_path
        # self.MODEL_CONFIDENCE_THRESHOLD = 0.5 # model confidence threshold; values below this threshold is discarded
        # self.MODEL_IOU_THRESHOLD = 0.2 #IoU threshold for the non-maximum suppression (NMS) algorithm
        self.model_name = 'yolo_nas_m'
        self.pretrained_weights = pretrained_weights
        self.data_prepared = False
        self.is_model_trained = False
        self.default_video_path = './fall.mp4'
        # specify whether the mode to train the model in: cpu, gpu, data parallel, etc
        # setup_device(multi_gpu='DP', num_gpus=1)
        # Define the directory where the dataset is stored and the class names
        self.dataset_params = {
            'data_dir': 'fall_dataset_v3',
            'train_images_dir': 'train/images',
            'train_labels_dir': 'train/labels',
            'val_images_dir': 'valid/images',
            'val_labels_dir': 'valid/labels',
            'test_images_dir': 'test/images',
            'test_labels_dir': 'test/labels',
            'classes': ['Fall-Detected']  # must specify the classes in the dataset
        }

        # Load the YOLO-NAS model from Super Gradients module using the specified architecture and the fine-tuned weights
        self.trained_model = self.get_model()

    def data_prep(self, display_sample=True):
        # This method instantiates a data loader which is responsible for loading data in batches, performing shuffling, and multiprocessing for faster load
        self.data_prepared = True
        self.train_dataloader = coco_detection_yolo_format_train(
            dataset_params={
                'data_dir': self.dataset_params['data_dir'],
                'images_dir': self.dataset_params['train_images_dir'],
                'labels_dir': self.dataset_params['train_labels_dir'],
                'classes': self.dataset_params['classes']
            },
            dataloader_params={
                'batch_size': 8,  # may need to lower this value if GPU runs out of memory
                'num_workers': 2
            }
        )

        self.val_dataloader = coco_detection_yolo_format_val(
            dataset_params={
                'data_dir': self.dataset_params['data_dir'],
                'images_dir': self.dataset_params['val_images_dir'],
                'labels_dir': self.dataset_params['val_labels_dir'],
                'classes': self.dataset_params['classes']
            },
            dataloader_params={
                'batch_size': 8,
                'num_workers': 2
            }
        )

        self.test_dataloader = coco_detection_yolo_format_val(
            dataset_params={
                'data_dir': self.dataset_params['data_dir'],
                'images_dir': self.dataset_params['test_images_dir'],
                'labels_dir': self.dataset_params['test_labels_dir'],
                'classes': self.dataset_params['classes']
            },
            dataloader_params={
                'batch_size': 8,
                'num_workers': 2
            }
        )

        # if display_sample:
        #     # visualize the data loaded using data loader where data augmentation is applied
        #     self.val_dataloader.dataset.plot(plot_transformed_data=True)

    def fine_tune(self, train_params=None, epochs=100):
        # This method fine tune the loaded YOLO-NAS model on the loaded data
        # Create a dataloaders if it has not already been created
        if not self.data_prepared:
            self.data_prep(display_sample=False)

        # Define training hyper-parameters
        train_params = {
            'silent_mode': False,  # enabling silent mode during trainining
            'average_best_models': True,
            'warmup_mode': 'LinearEpochLRWarmup',
            'warmup_initial_lr': 1e-6,
            'lr_warmup_epochs': 3,
            'initial_lr': 5e-4,
            'lr_mode': 'cosine',
            'cosine_final_lr_ratio': 0.1,
            'optimizer': 'Adam',
            'optimizer_params': {'weight_decay': 0.0001},
            'zero_weight_decay_on_bias_and_bn': True,
            'ema': True,
            'ema_params': {'decay': 0.9, 'decay_type': 'threshold'},
            'max_epochs': epochs,
            'mixed_precision': False,
            'loss': PPYoloELoss(use_static_assigner=False, num_classes=len(self.dataset_params['classes']), reg_max=16),
            # 'valid_metrics_list': [DetectionMetrics_050(
            #     score_thres=0.1,
            #     top_k_predictions=300,
            #     num_cls=len(self.dataset_params['classes']),
            #     normalize_targets=True,
            #     post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.1, nms_top_k=1000, max_predictions=300, nms_threshold=0.7)
            # )],
            'metric_to_watch': 'PPYoloELoss/loss_iou'
        }

        # Create a Trainer which is responsible for training, evaluation, and saving checkpoints
        # multi_gpu is defined as AUTO which will use GPU when available
        # the model parameters will be saved to the directory given in the argument when instantiating the Trainer class
        self.trainer = Trainer(experiment_name=self.EXPERIMENT_NAME, ckpt_root_dir=self.CHECKPOINT_DIR)

        # Begin training the model using the specified loaded model, data loaders, and training parameters
        # Load the YOLO-NAS model from Super Gradients module using the specified architecture
        model = models.get(self.model_name,
                           num_classes=len(self.dataset_params['classes']),
                           pretrained_weights=self.pretrained_weights
                           )
        print(f"The YOLONAS model is being trained...")
        self.trainer.train(model=model, training_params=train_params, train_loader=self.train_dataloader,
                           valid_loader=self.val_dataloader)
        print(f"Training Finished")
        self.is_model_trained = True

    def get_model(self):
        # Train the model if it has not been traineTRAINED_MODEL_PATHmodel path is not provided
        if self.is_model_trained == False and self.TRAINED_MODEL_PATH == None:
            self.fine_tune(epochs=50)

        # Load the fine-tuned model and return it
        print(f"Loading the YOLO-NAS model...")
        self.trained_model = models.get(
            self.model_name,
            num_classes=len(self.dataset_params['classes']),
            checkpoint_path=self.TRAINED_MODEL_PATH  # must specified where the trained model parameters is located
        )
        # Make the model run on GPU if available to save inference and training time
        print(f"Finished!")
        if torch.cuda.is_available():
            self.trained_model = self.trained_model.cuda()
        return self.trained_model

    def evaluate(self, confidence_threshold=0.50):
        # this method evaluate the trained YOLONAS model on the test dataset which was loaded using the data loader
        # check if the dataloader for the test data is instantiated, 
        if self.data_prepared == False:
            self.data_prep()

        # The model is not trained, and the trained weights is not provided, 
        if self.TRAINED_MODEL_PATH is None and self.is_model_trained == False:
            print(f"Error, cannot evaluate an empty model")
            return

        # Create a trainer who is responsible for evaluating the model given the data loaders
        trainer = Trainer(experiment_name=self.EXPERIMENT_NAME, ckpt_root_dir=self.CHECKPOINT_DIR)

        # Evaluate the model
        print(f"Beginning model evaluation...")
        results = trainer.test(model=self.trained_model,
                               test_loader=self.test_dataloader,
                               test_metrics_list=DetectionMetrics_050(score_thres=confidence_threshold, # minimum confidence level for a detection to be considered valid
                                                                      top_k_predictions=300,
                                                                      num_cls=len(self.dataset_params['classes']),
                                                                      normalize_targets=True,
                                                                      post_prediction_callback=PPYoloEPostPredictionCallback(
                                                                          score_threshold=confidence_threshold,
                                                                          nms_top_k=1000,
                                                                          max_predictions=300,
                                                                          nms_threshold=0.5 # non maximum suppression threshold; ensure each object is detected only once
                                                                          ) 
                                                                      )
                               )

        return results

    def infer(self, input_path, output_path, save_result=True, iou_threshold=0.5, confidence_threshold=0.35):
        # This method make inferences on the given fall data; accepts URL and directories to images or video
        # Get the trained model
        trained_model = self.get_model()

        # We want to use cuda if available to speed up inference.
        if torch.cuda.is_available():
            self.trained_model = self.trained_model.cuda()

        print(f"Making an inference on the provided input...")
        # predictions = self.model.predict(image_or_video_path, iou=self.MODEL_IOU_THRESHOLD, conf=self.MODEL_CONFIDENCE_THRESHOLD)
        predictions = self.trained_model.predict(input_path, iou=iou_threshold, conf=confidence_threshold)
        # arguments of predictions.save() are output_path, box_thickness, show_confidence, color_mapping, class_names
        if save_result:
            predictions.save(output_path)
        print(f"The detection is saved to {output_path} directory")
        # return the prediction object in which we can call .show() or .save() on
        return predictions
    
    def detect_video(self, video_path, confidence_threshold=0.35):
        # This method takes a video and performs fall detection, returning the frames with deteted falls
        # check if the video file can be opened and read
        cap = cv2.VideoCapture(video_path)
        frames = []
        if not cap.isOpened():
            print("error, cannot open the video")
            return []
        
        # read all the frames into a list to be processed
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # process every 2 frames instead to reduce computation overload without major loss in accuracy
            if frame_count % 2 == 0:
                frames.append(frame)
            frame_count += 1        
        cap.release()

        # use the model to make inferences on these testing images
        images_predictions = self.trained_model.predict(frames, skip_image_resizing=True)
        
        # list for storing the image prediction objects where the fall confidence is greater than the given confidece threshol
        fall_image_predictions = []
        for frame_number, image_prediction in enumerate(images_predictions):
            class_names = image_prediction.class_names
            labels = image_prediction.prediction.labels
            confidence = image_prediction.prediction.confidence
            bboxes = image_prediction.prediction.bboxes_xyxy
            for i, (label, conf, bbox) in enumerate(zip(labels, confidence, bboxes)):
                # if the model's confidence is greater than the given threshold, identify this frame as a fall and save the image prediction object
                if conf > confidence_threshold: 
                    print(f"{class_names[int(label)]} on frame # {frame_number} with confidence of {conf*100:.2f}%")
                    fall_image_predictions.append(image_prediction)
        
        # return the fall image prediction objects in which we can call .save() or .show() to save the frame or show the frame
        return fall_image_predictions
            

def send_alert():
    # Sends an alert to the user letting them know that a fall has occured
    pass


