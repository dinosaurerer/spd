import os.path
import time
import numpy as np
import torch
from pathlib import Path
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from models.common import DetectMultiBackend_YOLOv5
from yolocode.yolov5.YOLOv5Thread import YOLOv5Thread
from yolocode.yolov5.utils.dataloaders import LoadImages, LoadScreenshots, LoadStreams
from ultralytics.utils.plotting import Annotator, colors
from yolocode.yolov5.utils.general import (
    Profile,
    check_file,
    check_img_size,
    cv2,
    increment_path,
    non_max_suppression,
    scale_boxes,

)
from yolocode.yolov5.utils.segment.general import process_mask, process_mask_native
from yolocode.yolov5.utils.torch_utils import select_device


class YOLOv5SegThread(YOLOv5Thread):

    def __init__(self):
        super(YOLOv5SegThread, self).__init__()
        self.project = 'runs/segment'
        self.retina_masks = False

    @torch.no_grad()
    def detect(self, device, bs, is_folder_last=False):
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        # seen 表示图片计数
        datasets = iter(self.dataset)
        count = 0  # run location frame
        start_time = time.time()  # used to calculate the frame rate
        while True:
            if self.stop_dtc:
                if self.is_folder and not is_folder_last:
                    break
                self.send_msg.emit('Stop Detection')
                # --- 发送图片和表格结果 --- #
                self.send_result_picture.emit(self.results_picture)  # 发送图片结果
                for key, value in self.results_picture.items():
                    self.results_table.append([key, str(value)])
                self.results_picture = dict()
                self.send_result_table.emit(self.results_table)  # 发送表格结果
                self.results_table = list()
                # --- 发送图片和表格结果 --- #
                # 释放资源
                if hasattr(self.dataset, 'threads'):
                    for thread in self.dataset.threads:
                        if thread.is_alive():
                            thread.join(timeout=1)  # Add timeout
                if hasattr(self.dataset, 'cap') and self.dataset.cap is not None:
                    self.dataset.cap.release()
                cv2.destroyAllWindows()
                if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                    self.vid_writer[-1].release()
                break
            #  判断是否更换模型
            if self.current_model_name != self.new_model_name:
                weights = self.current_model_name
                data = self.data
                self.send_msg.emit(f'Loading Model: {os.path.basename(weights)}')
                self.model = DetectMultiBackend_YOLOv5(weights, device=device, dnn=False, data=data, fp16=False)
                stride, names, pt = self.model.stride, self.model.names, self.model.pt
                imgsz = check_img_size(self.imgsz, s=stride)  # check image size
                self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz))  # warmup
                seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
                self.current_model_name = self.new_model_name
            # 开始推理
            if self.is_continue:
                if self.is_file:
                    self.send_msg.emit("Detecting File: {}".format(os.path.basename(self.source)))
                elif self.webcam and not self.is_url:
                    self.send_msg.emit("Detecting Webcam: Camera_{}".format(self.source))
                elif self.is_folder:
                    self.send_msg.emit("Detecting Folder: {}".format(os.path.dirname(self.source[0])))
                elif self.is_url:
                    self.send_msg.emit("Detecting URL: {}".format(self.source))
                else:
                    self.send_msg.emit("Detecting: {}".format(self.source))
                path, im, im0s, self.vid_cap, s = next(datasets)
                # 原始图片送入 input框
                self.send_input.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                count += 1
                percent = 0  # 进度条
                # 处理processBar
                if self.vid_cap:
                    percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.progress_value)
                    self.send_progress.emit(percent)
                else:
                    percent = self.progress_value
                if count % 5 == 0 and count >= 5:  # Calculate the frame rate every 5 frames
                    self.send_fps.emit(str(int(5 / (time.time() - start_time))))
                    start_time = time.time()

                with dt[0]:
                    im = torch.from_numpy(im).to(self.model.device)
                    im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                # Inference
                with dt[1]:
                    pred, proto = self.model(im, augment=False, visualize=False)[:2]

                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, False,
                                               max_det=self.max_det, nm=32)

                # Process predictions
                for i, det in enumerate(pred):
                    seen += 1
                    # per image
                    if self.webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(self.dataset, "frame", 0)

                    self.file_path = p = Path(p)  # to Path
                    if self.save_res:
                        save_path = str(self.save_path / p.name)  # im.jpg
                        self.res_path = save_path
                    annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

                    # 类别数量 目标数量
                    class_nums = 0
                    target_nums = 0
                    if len(det):
                        if self.retina_masks:
                            # scale bbox first the crop masks
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                                     im0.shape).round()  # rescale boxes to im0 size
                            masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                        else:
                            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                                     im0.shape).round()  # rescale boxes to im0 size

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            class_nums += 1
                            target_nums += int(n)
                            if self.names[int(c)] in self.labels_dict:
                                self.labels_dict[self.names[int(c)]] += int(n)
                            else:  # 第一次出现的类别
                                self.labels_dict[self.names[int(c)]] = int(n)
                        # Mask plotting
                        annotator.masks(
                            masks,
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(
                                0).contiguous() /
                                   255 if self.retina_masks else im[i])

                        # Write results
                        for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                            c = int(cls)  # integer class
                            label = self.names[c]
                            confidence = float(conf)
                            confidence_str = f"{confidence:.2f}"
                            c = int(cls)  # integer class
                            label = f"{self.names[c]} {conf:.2f}"
                            annotator.box_label(xyxy, label, color=colors(c, True))

                    # 发送结果
                    im0 = annotator.result()
                    self.send_output.emit(im0)  # 输出图片
                    self.send_class_num.emit(class_nums)
                    self.send_target_num.emit(target_nums)
                    self.results_picture = self.labels_dict

                    if self.save_res:
                        if self.dataset.mode == "image":
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if self.vid_path[i] != save_path:  # new video
                                self.vid_path[i] = save_path
                                if isinstance(self.vid_writer[i], cv2.VideoWriter):
                                    self.vid_writer[i].release()  # release previous video writer
                                if self.vid_cap:  # video
                                    fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path = str(
                                    Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                                self.vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                                                     (w, h))
                            self.vid_writer[i].write(im0)

                    if self.speed_thres != 0:
                        time.sleep(self.speed_thres / 1000)  # delay , ms

                if self.is_folder and not is_folder_last:
                    # 判断当前是否为视频
                    if self.file_path and self.file_path.suffix[1:] in VID_FORMATS and percent != self.progress_value:
                        continue
                    break

                if percent == self.progress_value and not self.webcam:
                    self.send_progress.emit(0)
                    self.send_msg.emit('Finish Detection')
                    # --- 发送图片和表格结果 --- #
                    self.send_result_picture.emit(self.results_picture)  # 发送图片结果
                    for key, value in self.results_picture.items():
                        self.results_table.append([key, str(value)])
                    self.results_picture = dict()
                    self.send_result_table.emit(self.results_table)  # 发送表格结果
                    self.results_table = list()
                    # --- 发送图片和表格结果 --- #
                    self.res_status = True
                    if self.vid_cap is not None:
                        self.vid_cap.release()
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # release final video writer
                    break
