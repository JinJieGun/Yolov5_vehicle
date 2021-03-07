import argparse
import time
from pathlib import Path
import shapely  # conda install shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path, xywh2xyxy, box_iou, bbox_iou
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

PATH = '../output'
sigma_iou = 0.90  # iou阈值
timethre = 9
under_floor_car_park_area = [964, 25, 1920, 134, 567, 260, 1920, 892]  # 地下停车场停车区
fps = 30  # 每秒帧数
v_t = 10  # 判定违停的时间大小
capacity = np.zeros(100)  # 最大检测容量


def get_iou(a, b):
    '''
    :param a: box a [x0,y0,x1,y1,x2,y2,x3,y3]
    :param b: box b [x0,y0,x1,y1,x2,y2,x3,y3]
    :return: iou of bbox a and bbox b
    '''
    a, b = np.array(a), np.array(b)
    a = a.reshape(4, 2)
    poly1 = Polygon(a).convex_hull

    b = b.reshape(4, 2)
    poly2 = Polygon(b).convex_hull

    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            union_area = poly1.area + poly2.area - inter_area
            if union_area == 0:
                iou = 0
            else:
                iou = inter_area / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def iouCompare(Pre=[], Now=[], t_value=[], t_value_m=[]):
    for j in range(len(Now)):
        for i in range(len(Pre)):
            tmp = t_value.copy()
            iou = bbox_iou(Pre[i], Now[j])
            if iou >= sigma_iou:
                t_value[j] = tmp[i] + 1
                t_value_m[j] = 1
                break
            else:
                t_value[j] += 0


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    Pre, Now, objectListNowLine, xyxyInfo = [], [], [], []
    t_value, t_value_m = capacity, capacity
    f_num = 0  # 当前帧数

    for path, img, im0s, vid_cap in dataset:
        Now.clear()
        objectListNowLine.clear()
        xyxyInfo.clear()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = Path(path), '', im0s

            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                    infoDetail = '%s %d %d %d %d %.2f' % (
                        names[int(cls)], int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf)
                    if save_txt:  # Write to file
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if int(cls) == 1 or int(cls) == 2 or int(cls) == 3 or int(cls) == 5 or int(cls) == 7:
                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=(255, 0, 0), line_thickness=3)
                        Now.append(torch.tensor(xyxy).T)
                        objectListNowLine.append(infoDetail)
                        xyxyInfo.append(xyxy)
            # Print time (inference + NMS)
            # Stream results
            if view_img:
                cv2.namedWindow(str(p), 0)
                cv2.resizeWindow(str(p), 640, 480)
                cv2.imshow(str(p), im0)
                print("view window")
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps_1 = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps_1, (w, h))
                    vid_writer.write(im0)

        if f_num % fps == 0:
            iouCompare(Pre, Now, t_value, t_value_m)

        for mark in range(len(t_value_m)):
            if t_value_m[mark] == 0:
                t_value[mark] /= 2
        if f_num % (v_t * fps) == 0:
            time_h_str = time.strftime("%Y-%m-%d-%H", time.localtime())
            time_s_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            with open('logs/' + str(time_h_str) + '.txt', 'a+') as log:
                log.writelines(time_s_str + ' \n')
            for i in range(len(Now)):
                if t_value[i] >= timethre:
                    im0s = np.ascontiguousarray(im0s)
                    # obj_axis = [int(xyxyInfo[i][0]), int(xyxyInfo[i][1]),  int(xyxyInfo[i][2]),int(xyxyInfo[i][1]),
                    # int(xyxyInfo[i][0]),int(xyxyInfo[i][3]),  int(xyxyInfo[i][2]),int(xyxyInfo[i][3])] if get_iou(
                    # obj_axis,under_floor_car_park_area)==0:
                    cv2.rectangle(im0s, (int(xyxyInfo[i][0]), int(xyxyInfo[i][1])),
                                  (int(xyxyInfo[i][2]), int(xyxyInfo[i][3])), color=(0, 0, 255), thickness=3,
                                  lineType=cv2.LINE_AA)
                    with open('logs/' + str(time_h_str) + '.txt', 'a+') as log:
                        log.writelines(
                            "%d" % i + " 违停车辆：" + objectListNowLine[i] + '违停时间. (%.3fs)' % (t_value[i]) + '\n')
            cv2.imwrite("pics/img_save{}.png".format(time_s_str), np.float32(im0s))
            t_value, t_value_m = capacity, capacity

        if f_num % fps == 0:
            Pre = Now.copy()
        f_num += 1

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5x.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

