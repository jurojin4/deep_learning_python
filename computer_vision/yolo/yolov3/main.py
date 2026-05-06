from .trainer import YOLOV3Trainer

import argparse

def parse_tuple(s):
    tup = tuple(map(int, s.strip("()").split(",")))
    if len(tup) == 1:
        return tup[0], tup[0]
    else:
        return tup

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOV3 Trainer', add_help=add_help)
    parser.add_argument("--dataset_name", default="pascalvoc2012", type=str, help='datasets available: ["cifar10", "facedetection", "pascalvoc2007", "pascalvoc2012", "tiny-imagenet200"]')
    parser.add_argument("--dataset_path", default=None, type=str, help="directory where the dataset is")
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs during training")
    parser.add_argument("--image_size", default=(224, 224), type=parse_tuple, help="Image sizes")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size during training")
    parser.add_argument("--iou_threshold_overlap", default=None, type=float, help="IoU overlap limit between predicted bounding boxes")
    parser.add_argument("--confidence_threshold", default=None, type=float, help="Confidence limit to exceed by bounding boxes to be considerated according the model as correct")
    parser.add_argument("--warmup_epoch", default=1, type=int, help="Number of epochs where the model is not measured.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="learning rate during training")
    parser.add_argument("--milestones", default=[45], type=list, help="Epoch steps where learning rate decreasing lineary by 0.1")
    parser.add_argument("--detail", action="store_true", help="extra details for metrics")
    parser.add_argument("--no_measure", action="store_true", help="No metrics will be measured")
    parser.add_argument("--save", action="store_true", help="save model during training")
    parser.add_argument("--save_metric", default="loss", type=str, help="metric used for model save")
    parser.add_argument("--box_format", default="xywh", type=str, help="format of bounding boxes")
    parser.add_argument("--data_augmentation", action="store_true", help="Data augmentation")
    parser.add_argument("--delete", action="store_true", help="delete dataset if it has been downloaded")
    parser.add_argument("--weights_path", default=None, type=str, help="Path of the weights")
    parser.add_argument("--load_all", action="store_true", help="Boolean that loads completely are not the model")
    parser.add_argument("--experiment_name", default=None, type=str, help="Name of the experiment")
    parser.add_argument("--no_verbose", action="store_true", help="detailed training on terminal")
    return parser
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    trainer = YOLOV3Trainer(dataset_name=args.dataset_name,
                dataset_path=args.dataset_path,
                epochs=args.epochs,
                image_size=args.image_size,
                batch_size=args.batch_size,
                iou_threshold_overlap=args.iou_threshold_overlap,
                confidence_threshold=args.confidence_threshold,
                warmup_epoch=args.warmup_epoch,
                learning_rate=args.learning_rate,
                milestones=args.milestones,
                detail=args.detail,
                no_measure=args.no_measure,
                save=args.save,
                save_metric=args.save_metric,
                box_format=args.box_format,
                data_aug=args.data_augmentation,
                delete=args.delete,
                weights_path=args.weights_path,
                load_all=args.load_all,
                experiment_name=args.experiment_name,
                no_verbose=args.no_verbose)
    trainer()