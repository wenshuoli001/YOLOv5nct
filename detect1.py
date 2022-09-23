# YOLOv5 ğŸš€ by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

#导入Python库
import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()#得到当前文件的绝对路径：WindowsPath('C:/Users/Lee/Desktop/YOLO-swin transformer/yolov5-master/detect.py')
ROOT = FILE.parents[0]  # YOLOv5 root directory #得到整个YOLOv5项目的路径
if str(ROOT) not in sys.path:#判断需要导入的模块是否在上面的路径内
    sys.path.append(str(ROOT))  # 如果不在add ROOT to PATH，确保接下来导入能够成功
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative转化为相对路径：WindowsPath('.')

#导入路径下的模块（其他文件）
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
#导入完成后，接下来都是定义，直接跳过执行最下面的 if main

@torch.no_grad()
#定义run函数
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)默认模型是s
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    #先对参数中传入的source进行一系列判断
    source = str(source)#传入的source路径，并转化为字符串
    save_img = not nosave and not source.endswith('.txt')  # save inference images保存预测结果，可以通过传入参数改变，上面有nosave参数的输入，可以改成TRUE
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)#判断传入的路径是不是一个文件地址，suffix表示后缀,判断后缀是不是一个图像或者视频的格式
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))#判断传入的地址是不是一个网络流地址
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)#判断传入的source是不是一个数值，传入数字0表示打开电脑摄像头
    if is_url and is_file:#如果传入是网络流地址并且是个文件，那么下载该图片或者视频
        source = check_file(source)  # download

    # Directories建立保存结果的文件夹
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run增量路径，检测exp文件夹到几了，创建大一号的
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir如果save_txt=TRUE（输入参数时可调整）那么还会保存边界框信息到txt文件中

    # Load model加载模型权重
    device = select_device(device)#选择cuda
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)#选择多后端框架，判断使用pytorch还是tenserflow啥的
    stride, names, pt = model.stride, model.names, model.pt#读取模型参数
    imgsz = check_img_size(imgsz, s=stride)  # check image size判断图像是不是32的倍数

    # Dataloader加载待遇测的图片
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)#加载图像
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference执行模型推理计算
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup传入空白图片前馈传播使GPU工作
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]#dt用于存储时间
    for path, im, im0s, vid_cap, s in dataset:#遍历dataloder中的图片,一张一张的开始预测
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)#im是resize后的图片,从numpy格式转成tenser后放进device(gpu)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32 数字精度
        im /= 255  # 0 - 255 to 0.0 - 1.0 像素值归一化
        if len(im.shape) == 3:#判断im维度是否等于3,如果等于三,在前面加入batch维度
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False#visualize默认为FALSE,如果为TRUE推断过程中会保存图像特征图
        pred = model(im, augment=augment, visualize=visualize)#augment参数:传入时是否需要数据增强
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS#根据confidence和iou过滤边界框,一般会生成两万个左右的边界框
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)#max_det最大目标数,超过自动过滤,默认为一千,每个边界框输出为坐标+置信度+类别
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image遍历batch中的每个图片(i)和每个检测框(det)
            seen += 1
            if webcam:  # batch_size >= 1 取值
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg保存
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh获取原图大小后面有用
            imc = im0.copy() if save_crop else im0  # for save_crop判断是否需要裁减出检测框单独保存
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))#绘图工具,传入1原图2检测框粗细3标签名
            if len(det):#如果有边界框 画图
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()#将resize后的框坐标映射到原图上,因为要在原图上画

                # Print results
                for c in det[:, -1].unique():#遍历所有边界框,统计类别输出打印
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')#是否需要隐藏标签和置信度
                        annotator.box_label(xyxy, label, color=colors(c, True))#调用annotator中的box_label方法
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:#是否现实图片
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)#OpenCV函数保存图片
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results打印输出信息
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


#解析命令行参数的函数的定义 运行传入参数时使用%run detect.py --source data\\images\\bus.jpg
def parse_opt():
    #定义命令行中可以传入的参数，不传入的按默认处理
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'YOLOv5nct_voc100.pt', help='model path(s)')#预测用模型
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')#数据集
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')#传入网络的图像大小
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')#保留预测框的置信度阈值
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')#保留预测框的iou阈值
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')#是否现实图像,只需要命令行输入--view-img即可 如果是视频则会实时现实视频
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')#保留类别数,如果输入--classes 0 则只保留人的预测框
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')#执行nms强化,提升检测效果,只需要输入--agnostic-nms
    parser.add_argument('--augment', action='store_true', help='augmented inference')#执行图像增强,提升检测效果,只需要输入--augment
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')#结果保存路径
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()#opt变量存储参数信息，作为后面main函数的输入
    #对图像大小进行额外判断
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand如果传入只有一个640那么改成640*640
    print_args(vars(opt))#打印传入的参数信息
    return opt


#定义main函数，使用上面传入的参数opt 作为输入
def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))#检测是否满足requirement文件中的库的要求
    run(**vars(opt))#一系列操作都在run函数中，上面有定义


#导入完成后执行这个
if __name__ == "__main__":
    opt = parse_opt()#解析传入参数（命令行（右边窗口）中），上面有函数定义
    main(opt)#接下来执行main函数
