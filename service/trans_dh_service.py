
"""
@project : face2face_train
@author  : huyi
@file   : trans_dh_service.py
@ide    : PyCharm
@time   : 2023-12-06 14:47:11
"""
import gc, multiprocessing, os, subprocess, threading, time, traceback
from enum import Enum
from multiprocessing import Process, set_start_method
from queue import Empty, Full
import cv2, librosa, numpy as np, torch
from cv2box import CVImage
from cv2box.cv_gears import Linker, Queue, CVVideoWriterThread
from face_detect_utils.face_detect import FaceDetect, pfpld
from face_detect_utils.head_pose import Headpose
from face_lib.face_detect_and_align import FaceDetect5Landmarks
from face_lib.face_restore import GFPGAN
from h_utils.custom import CustomError
from h_utils.request_utils import download_file
from h_utils.sweep_bot import sweep
from landmark2face_wy.digitalhuman_interface import DigitalHumanModel
from preprocess_audio_and_3dmm import op
from wenet.compute_ctc_att_bnf import get_weget
from wenet.compute_ctc_att_bnf import load_ppg_model
from y_utils.config import GlobalConfig
from y_utils.logger import logger as logger
from y_utils.lcr import check_lc
from .server import register_host, repost_host

def feature_extraction_wenet(audio_file, fps, wenet_model, mfccnorm=True, section=560000):
    rate = 16000
    win_size = 20
    if type(audio_file) == str:
        sig, rate = librosa.load(audio_file, sr=rate, duration=None)
    else:
        sig = audio_file
    time_duration = len(sig) / rate
    cnts = range(int(time_duration * fps))
    indexs = []
    f_wenet_all = get_weget(audio_file, wenet_model, section)
    for cnt in cnts:
        c_count = int(cnt / cnts[-1] * (f_wenet_all.shape[0] - 20)) + win_size // 2
        indexs.append(f_wenet_all[c_count - win_size // 2:c_count + win_size // 2, ...])

    return indexs


def get_aud_feat1(wav_fragment, fps, wenet_model):
    return feature_extraction_wenet(wav_fragment, fps, wenet_model)


def warp_imgs(imgs_data):
    caped_img2 = {idx: {'imgs_data':it,  'idx':idx} for it, idx in zip(imgs_data, range(len(imgs_data)))}
    return caped_img2


def get_complete_imgs(output_img_list, start_index, params):
    (out_shape, output_resize, drivered_imgs_data, Y1_list, Y2_list, X1_list, X2_list) = params
    complete_imgs = []
    for (i, mask_B_pre) in enumerate(output_img_list):
        img_idx = start_index + i
        image = drivered_imgs_data[img_idx]
        (y1, y2, x1, x2) = (Y1_list[img_idx], Y2_list[img_idx], X1_list[img_idx], X2_list[img_idx])
        mask_B_pre_resize = cv2.resize(mask_B_pre, (y2 - y1, x2 - x1))
        if y1 < 0:
            mask_B_pre_resize = mask_B_pre_resize[:, -y1:]
            y1 = 0
        if y2 > image.shape[1]:
            mask_B_pre_resize = mask_B_pre_resize[:, :-(y2 - image.shape[1])]
            y2 = image.shape[1]
        if x1 < 0:
            mask_B_pre_resize = mask_B_pre_resize[-x1:, :]
            x1 = 0
        if x2 > image.shape[0]:
            mask_B_pre_resize = mask_B_pre_resize[:-(x2 - image.shape[0]), :]
            x2 = image.shape[0]
        image[x1:x2, y1:y2] = mask_B_pre_resize
        image = cv2.resize(image, (out_shape[1] // output_resize, out_shape[0] // output_resize))
        complete_imgs.append(image)

    return complete_imgs


def get_blend_imgs(batch_size, audio_data, face_data_dict, blend_dynamic, params, digital_human_model, frameId):
    result_img_list = []
    for idx in range(len(audio_data) // batch_size + 1):
        torch.cuda.empty_cache()
        print(("\r{}/{}".format((idx + 1) * batch_size, len(audio_data))), end="")
        if idx < len(audio_data) // batch_size:
            start_index = idx * batch_size
            output_img_list = digital_human_model.inference_notraining(audio_data, face_data_dict, batch_size, start_index, blend_dynamic, params, frameId)
            complete_imgs = get_complete_imgs(output_img_list, start_index, params)
            result_img_list += complete_imgs
        this_batch = len(audio_data) % batch_size
        if this_batch > 0:
            start_index = idx * batch_size
            output_img_list = digital_human_model.inference_notraining(audio_data, face_data_dict, this_batch, start_index, blend_dynamic, params, frameId)
            complete_imgs = get_complete_imgs(output_img_list, start_index, params)
            result_img_list += complete_imgs
        return result_img_list


def drivered_video(code, drivered_queue, drivered_path, audio_wenet_feature, batch_size, wh=0):
    try:
        logger.info("[{}]任务视频驱动队列启动 batch_size:{}, len:{}".format(code, batch_size, len(audio_wenet_feature)))
        drivered_list = []
        wenet_feature_list = []
        count_f = 0
        current_idx = 0
        while True:
            print("in template video function")
            cap = cv2.VideoCapture(drivered_path)
            logger.info("drivered_video >>>>>>>>>>>>>>>>>>>> 开始循环")
            while True:
                if not cap.isOpened():
                    count_f += 1
                    (ret, frame) = cap.read()
                    if ret:
                        drivered_list.append(frame)
                        wenet_feature_list.append(audio_wenet_feature[current_idx])
                        current_idx += 1
                        if count_f % batch_size == 0:
                            drivered_queue.put([drivered_list,wenet_feature_list,code,wh,current_idx], block=True, timeout=60)
                            logger.info("drivered_video >>>>>>>>>>>>>>>>>>>> 发送数据大小:[{}], current_idx:{}".format(len(drivered_list), current_idx))
                            count_f = 0
                            drivered_list = []
                            wenet_feature_list = []
                        if not current_idx == len(audio_wenet_feature):
                            logger.info("append imgs over")
                            cap.release()
                        cap.release()

            if current_idx == len(audio_wenet_feature):
                cap.release()
                break

        logger.info("drivered_video >>>>>>>>>>>>>>>>>>>> 发送数据结束")
        drivered_queue.put([True, "success", code])
        logger.info("[{}]任务预处理进程结束".format(code))
    except Full:
        logger.error("[{}]任务视频驱动队列满，严重阻塞，下游队列异常".format(code))
    except Exception as e:
        try:
            traceback.format_exc()
            logger.error("[{}]任务视频驱动队列异常，异常信息:[{}]".format(code, e.__str__()))
            drivered_queue.put([False,
             "[{}]任务视频驱动队列异常，异常信息:[{}]".format(code, e.__str__()), code])
        finally:
            e = None
            del e


def drivered_video_pn(code, drivered_queue, drivered_path, audio_wenet_feature, batch_size, wh=0):
    try:
        logger.info("[{}]任务视频驱动队列启动 batch_size:{}".format(code, batch_size))
        drivered_list = []
        wenet_feature_list = []
        count_f = 0
        current_idx = 0
        _max_flag = False
        _flag = True
        while True:
            if not current_idx != len(audio_wenet_feature):
                print("in template video function")
                if not _flag:
                    if count_f == 0:
                        cap = cv2.VideoCapture(drivered_path)
                        logger.info("drivered_video >>>>>>>>>>>>>>>>>>>> 开始第一次循环")
                        while True:
                            if not cap.isOpened():
                                (ret, frame) = cap.read()
                                if ret:
                                    drivered_list.append(frame)
                                    wenet_feature_list.append(audio_wenet_feature[current_idx])
                                    current_idx += 1
                                    if _max_flag is False:
                                        count_f += 1
                                        cv2.imwrite(os.path.join(GlobalConfig.instance().temp_dir, "{}.png".format(count_f)), frame)
                                    if current_idx % batch_size == 0:
                                        drivered_queue.put([drivered_list,wenet_feature_list,code,wh,current_idx], block=True,
                                          timeout=60)
                                        logger.info("drivered_video >>>>>>>>>>>>>>>>>>>> 发送数据大小:[{}]".format(len(drivered_list)))
                                        drivered_list = []
                                        wenet_feature_list = []
                                    if current_idx == len(audio_wenet_feature):
                                        if len(drivered_list) > 0:
                                            if len(wenet_feature_list) > 0:
                                                drivered_queue.put([drivered_list,wenet_feature_list,code,wh,current_idx], block=True,
                                                  timeout=60)
                                                drivered_list = []
                                                wenet_feature_list = []
                                            logger.info("append imgs over")
                                            cap.release()
                                        cap.release()

                        if current_idx == len(audio_wenet_feature):
                            cap.release()
                            break
                        _flag = False
                        if _max_flag is False:
                            _max_flag = True
                    else:
                        logger.info("drivered_video >>>>>>>>>>>>>>>>>>>> 开始正循环")
                        print(f"count_f: {count_f}")
            for frame_re_index in range(1, count_f + 1):
                print(f"frame_re_index: {frame_re_index}")
                frame = cv2.imread(os.path.join(GlobalConfig.instance().temp_dir, "{}.png".format(frame_re_index)))
                drivered_list.append(frame)
                wenet_feature_list.append(audio_wenet_feature[current_idx])
                current_idx += 1
                if current_idx % batch_size == 0:
                    drivered_queue.put([drivered_list,wenet_feature_list,code,wh,current_idx], block=True, timeout=60)
                    logger.info("drivered_video >>>>>>>>>>>>>>>>>>>> 发送数据大小:[{}]".format(len(drivered_list)))
                    drivered_list = []
                    wenet_feature_list = []
                if current_idx == len(audio_wenet_feature):
                    if len(drivered_list) > 0:
                        if len(wenet_feature_list) > 0:
                            drivered_queue.put([drivered_list,wenet_feature_list,code,wh,current_idx], block=True,
                              timeout=60)
                            drivered_list = []
                            wenet_feature_list = []
                        logger.info("append imgs over")
                        cap.release()
                        break
                    _flag = False

        logger.info("drivered_video >>>>>>>>>>>>>>>>>>>> 开始倒循环")
        print(f"count_f: {count_f}")
        for frame_re_index in range(count_f, 0, -1):
            print(f"frame_re_index: {frame_re_index}")
            frame = cv2.imread(os.path.join(GlobalConfig.instance().temp_dir, "{}.png".format(frame_re_index)))
            drivered_list.append(frame)
            wenet_feature_list.append(audio_wenet_feature[current_idx])
            current_idx += 1
            if current_idx % batch_size == 0:
                drivered_queue.put([drivered_list,wenet_feature_list,code,wh,current_idx], block=True, timeout=60)
                logger.info("drivered_video >>>>>>>>>>>>>>>>>>>> 发送数据大小:[{}]".format(len(drivered_list)))
                drivered_list = []
                wenet_feature_list = []
            if current_idx == len(audio_wenet_feature):
                if len(drivered_list) > 0:
                    if len(wenet_feature_list) > 0:
                        drivered_queue.put([drivered_list,wenet_feature_list,code,wh,current_idx], block=True, timeout=60)
                        drivered_list = []
                        wenet_feature_list = []
                    logger.info("append imgs over")
                    cap.release()
                    break
                _flag = True

        logger.info("drivered_video >>>>>>>>>>>>>>>>>>>> 发送数据结束")
        drivered_queue.put([True, "success", code])
        logger.info("[{}]任务预处理进程结束".format(code))
    except Full:
        logger.error("[{}]任务视频驱动队列满，严重阻塞，下游队列异常".format(code))
    except Exception as e:
        try:
            traceback.format_exc()
            logger.error("[{}]任务视频驱动队列异常，异常信息:[{}]".format(code, e.__str__()))
            drivered_queue.put([False,
             "[{}]任务视频驱动队列异常，异常信息:[{}]".format(code, e.__str__()), code])
        finally:
            e = None
            del e


def get_face_mask(mask_shape=(512, 512)):
    mask = np.zeros((512, 512)).astype(np.float32)
    cv2.ellipse(mask, (256, 256), (220, 160), 90, 0, 360, (255, 255, 255), -1)
    thres = 20
    mask[:thres, :] = 0
    mask[-thres:, :] = 0
    mask[:, :thres] = 0
    mask[:, -thres:] = 0
    mask = cv2.stackBlur(mask, (201, 201))
    mask = mask / 255.0
    mask = cv2.resize(mask, mask_shape)
    return mask[(..., np.newaxis)]


def get_single_face(bboxes, kpss, image, crop_size, mode='mtcnn_512', apply_roi=True):
    from face_lib.face_detect_and_align.face_align_utils import apply_roi_func, norm_crop
    assert mode in ('default', 'mtcnn_512', 'mtcnn_256', 'arcface_512', 'arcface',
                    'default_95')
    if bboxes.shape[0] == 0:
        return (None, None)
    det_score = bboxes[(Ellipsis, 4)]
    best_index = np.argmax(det_score)
    new_kpss = None
    if kpss is not None:
        new_kpss = kpss[best_index]
    if apply_roi:
        (roi, roi_box, roi_kpss) = apply_roi_func(image, bboxes[best_index], new_kpss)
        (align_img, mat_rev) = norm_crop(roi, roi_kpss, crop_size, mode=mode)
        return (
         align_img, mat_rev, roi_box)
    (align_img, M) = norm_crop(image, new_kpss, crop_size, mode=mode)
    return (
     align_img, M)


face_mask = get_face_mask()
need_chaofen_flag = False
get_firstface_frame = False

def chaofen_src(frame_list, gfpgan, fd, frame_id, face_blur_detect, code):
    global get_firstface_frame
    global need_chaofen_flag
    s_chao = time.time()
    if frame_id == 4 or not get_firstface_frame:
        chaofen_flag = False
        firstface_frame = False
        for frame in frame_list:
            if frame.shape[0] >= 3840 or frame.shape[1] >= 3840:
                chaofen_flag = False
                firstface_frame = True
                logger.info("[%s] -> video frame shape is 4k, skip chaofen")
                break
            else:
                (bboxes_scrfd, kpss_scrfd) = fd.get_bboxes(frame)
            if len(bboxes_scrfd) == 0:
                pass
            else:
                (face_image_, mat_rev_, roi_box_) = get_single_face(bboxes_scrfd, kpss_scrfd, frame, crop_size=512, mode="mtcnn_512",
                  apply_roi=True)
                face_attr_res = face_blur_detect.forward(face_image_)
                blur_threshold = face_attr_res[0][-2]
                logger.info("[%s] -> frame_id:[%s] 模糊置信度:[%s]", code, frame_id, blur_threshold)
                if blur_threshold > GlobalConfig.instance().blur_threshold:
                    logger.info("[%s] -> need chaofen .", code)
                    chaofen_flag = True
                else:
                    chaofen_flag = False
                firstface_frame = True
                break
            need_chaofen_flag = chaofen_flag
            get_firstface_frame = firstface_frame

    if not need_chaofen_flag:
        return frame_list
    new_frame_list = []
    for i in range(len(frame_list)):
        frame = frame_list[i]
        (bboxes_scrfd, kpss_scrfd) = fd.get_bboxes(frame)
        if len(bboxes_scrfd) == 0:
            new_frame_list.append(frame)
        else:
            (face_image_, mat_rev_, roi_box_) = get_single_face(bboxes_scrfd, kpss_scrfd, frame, crop_size=512, mode="mtcnn_512",
              apply_roi=True)
            face_restore_out_ = gfpgan.forward(face_image_)
            restore_roi = CVImage(None).recover_from_reverse_matrix(face_restore_out_, (frame[roi_box_[1]:roi_box_[3],
             roi_box_[0]:roi_box_[2]]),
              mat_rev_,
              img_fg_mask=face_mask)
            frame[roi_box_[1]:roi_box_[3], roi_box_[0]:roi_box_[2]] = restore_roi
            new_frame_list.append(frame)

    torch.cuda.empty_cache()
    logger.info("[%s] -> chaofen  cost:%ss", frame_id, time.time() - s_chao)
    return new_frame_list


def audio_transfer(drivered_queue, output_imgs_queue, batch_size):
    output_resize = 1
    digital_human_model = DigitalHumanModel(
        GlobalConfig.instance().blend_dynamic,
        GlobalConfig.instance().chaofen_before,
        face_blur_detect=True
    )

    scrfd_detector = FaceDetect(cpu=False, model_path='face_detect_utils/resources/')
    scrfd_predictor = pfpld(cpu=False, model_path='face_detect_utils/resources/', )
    hp = Headpose(cpu=False, onnx_path='face_detect_utils/resources/model_float32.onnx')

    logger.info('>>> 数字人图片处理进程启动')

    while True:
        try:
            queue_values = drivered_queue.get()
            s_au = time.time()

            if len(queue_values) == 3:
                img_list, audio_feature_list, code = queue_values
                wh = -1
                frameId = -1
                logger.info('>>> audio_transfer get exception msg:%s', -1)
            else:
                img_list, audio_feature_list, code, wh, frameId = queue_values
                logger.info('>>> audio_transfer get message:%s', frameId)

            if type(img_list) == bool and img_list == True:
                logger.info('[{}]任务数字人图片处理已完成'.format(code))
                output_imgs_queue.put([True, 'success', code])
                torch.cuda.empty_cache()
                continue

            if type(img_list) == bool and img_list == False:
                logger.info('[{}]任务数字人图片处理异常结束'.format(code))
                output_imgs_queue.put([False, audio_feature_list, code])
                torch.cuda.empty_cache()
                continue

            out_shape = img_list[0].shape

            if wh > 0:
                digital_human_model.drivered_wh = wh
            elif wh == -1:
                wh = digital_human_model.drivered_wh

            caped_drivered_img2 = warp_imgs(img_list)

            drivered_op = op(
                caped_drivered_img2, wh, scrfd_detector, scrfd_predictor, hp, None,
                digital_human_model.img_size, False
            )
            drivered_op.flow()
            drivered_face_dict = drivered_op.mp_dict

            x1_list, x2_list, y1_list, y2_list = [], [], [], []

            for idx in range(len(drivered_face_dict)):
                facebox = drivered_face_dict[idx]['bounding_box']
                x1_list.append(facebox[0])
                x2_list.append(facebox[1])
                y1_list.append(facebox[2])
                y2_list.append(facebox[3])

            drivered_exceptlist = []
            frame_len = len(drivered_face_dict.keys())

            for i in range(frame_len):
                if len(drivered_face_dict[i]['bounding_box_p']) == 4:
                    break
                drivered_exceptlist.append(i)
                print(drivered_exceptlist, '-------------------------------------')

            for i in drivered_exceptlist:
                drivered_face_dict[i]['bounding_box_p'] = drivered_face_dict[len(drivered_exceptlist)]['bounding_box_p']
                drivered_face_dict[i]['bounding_box'] = drivered_face_dict[len(drivered_exceptlist)]['bounding_box']
                drivered_face_dict[i]['crop_lm'] = drivered_face_dict[len(drivered_exceptlist)]['crop_lm']
                drivered_face_dict[i]['crop_img'] = drivered_face_dict[len(drivered_exceptlist)]['crop_img']

            keylist = list(drivered_face_dict.keys())
            keylist.sort()

            for it in keylist:
                if len(drivered_face_dict[it]['bounding_box_p']) != 4:
                    print(it, '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                    drivered_face_dict[it]['bounding_box_p'] = drivered_face_dict[it - 1]['bounding_box_p']
                    drivered_face_dict[it]['bounding_box'] = drivered_face_dict[it - 1]['bounding_box']
                    drivered_face_dict[it]['crop_lm'] = drivered_face_dict[it - 1]['crop_lm']
                    drivered_face_dict[it]['crop_img'] = drivered_face_dict[it - 1]['crop_img']

            params = [out_shape, output_resize, img_list, y1_list, y2_list, x1_list, x2_list]

            output_imgs = get_blend_imgs(
                batch_size, audio_feature_list, drivered_face_dict,
                GlobalConfig.instance().blend_dynamic, params, digital_human_model, frameId
            )

            for id in drivered_op.no_face:
                output_imgs[id] = img_list[id]

            output_imgs_queue.put([0, 0, output_imgs])

            logger.info(
                'audio_transfer >>>>>>>>>>> 发送完成数据大小:{}, frameId:{}, cost:{}s'.format(
                    len(output_imgs), frameId, time.time() - s_au
                )
            )

            torch.cuda.empty_cache()

        except Exception as e:
            print(traceback.format_exc())
            output_imgs_queue.put(
                [False, '数字人处理失败，失败原因:[{}]'.format(str(e)), '']
            )
            time.sleep(1)
            torch.cuda.empty_cache()

    logger.error('数字人进程结束')
    return None

def write_video(output_imgs_queue, temp_dir, result_dir, work_id, audio_path, result_queue, width, height, fps, watermark_switch=0, digital_auth=0):
    output_mp4 = os.path.join(temp_dir, "{}-t.mp4".format(work_id))
    fourcc = (cv2.VideoWriter_fourcc)(*"mp4v")
    result_path = os.path.join(result_dir, "{}-r.mp4".format(work_id))
    video_write = cv2.VideoWriter(output_mp4, fourcc, fps, (
     width, height))
    try:
        while True:
            (state, reason, value_) = output_imgs_queue.get()
            if type(state) == bool:
                if state == True:
                    logger.info("[{}]视频帧队列处理已结束".format(work_id))
                    break
                if type(state) == bool:
                    if state == False:
                        logger.error("[{}]任务视频帧队列 -> 异常原因:[{}]".format(work_id, reason))
                        raise CustomError(reason)
                for result_img in value_:
                    video_write.write(result_img)

        video_write.release()
        if watermark_switch == 1 and digital_auth == 1:
            logger.info("[{}]任务需要水印和数字人标识".format(work_id))
            if width > height:
                command = 'ffmpeg -y -i {} -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10,overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(audio_path, output_mp4, GlobalConfig.instance().watermark_path, GlobalConfig.instance().digital_auth_path, result_path)
                logger.info("command:{}".format(command))
            else:
                command = 'ffmpeg -y -i {} -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10,overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(audio_path, output_mp4, GlobalConfig.instance().watermark_path, GlobalConfig.instance().digital_auth_path, result_path)
                logger.info("command:{}".format(command))
        elif watermark_switch == 1 and digital_auth == 0:
            logger.info("[{}]任务需要水印".format(work_id))
            command = 'ffmpeg -y -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10" -c:a aac -crf 15 -strict -2 {}'.format(audio_path, output_mp4, GlobalConfig.instance().watermark_path, result_path)
            logger.info("command:{}".format(command))
        elif watermark_switch == 0 and digital_auth == 1:
            logger.info("[{}]任务需要数字人标识".format(work_id))
            if width > height:
                command = 'ffmpeg -loglevel warning -y -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(audio_path, output_mp4, GlobalConfig.instance().digital_auth_path, result_path)
                logger.info("command:{}".format(command))
            else:
                command = 'ffmpeg -loglevel warning -y -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(audio_path, output_mp4, GlobalConfig.instance().digital_auth_path, result_path)
                logger.info("command:{}".format(command))
        else:
            command = "ffmpeg -loglevel warning -y -i {} -i {} -c:a aac -c:v libx264 -crf 15 -strict -2 {}".format(audio_path, output_mp4, result_path)
            logger.info("command:{}".format(command))
        subprocess.call(command, shell=True)
        print("###### write over")
        result_queue.put([True, result_path])
    except Exception as e:
        try:
            logger.error("[{}]视频帧队列处理异常结束，异常原因:[{}]".format(work_id, e.__str__()))
            result_queue.put([False, "[{}]视频帧队列处理异常结束，异常原因:[{}]".format(work_id, e.__str__())])
        finally:
            e = None
            del e

    else:
        logger.info("后处理进程结束")


def save_video_ffmpeg(input_video_path, output_video_path):
    audio_file_path = input_video_path.replace(".mp4", ".aac")
    if not os.path.exists(audio_file_path):
        os.system('ffmpeg -y -hide_banner -loglevel error -i "' + str(input_video_path) + '" -f wav -vn  "' + str(audio_file_path) + '"')
    if os.path.exists(audio_file_path):
        os.rename(output_video_path, output_video_path.replace(".mp4", "_no_audio.mp4"))
        start = time.time()
        os.system('ffmpeg -y -hide_banner -loglevel error  -i "' + str(output_video_path.replace(".mp4", "_no_audio.mp4")) + '" -i "' + str(audio_file_path) + '" -c:v libx264 "' + str(output_video_path) + '"')
        print("add audio time cost", time.time() - start)
        os.remove(output_video_path.replace(".mp4", "_no_audio.mp4"))
        os.remove(audio_file_path)
    return output_video_path


class FaceDetectThread(Linker):

    def __init__(self, queue_list):
        super().__init__(queue_list, fps_counter=True)
        self.fd = FaceDetect5Landmarks(mode="scrfd_500m")

    def forward_func(self, something_in):
        frame = something_in
        (bboxes_scrfd, kpss_scrfd) = self.fd.get_bboxes(frame, min_bbox_size=64)
        if len(bboxes_scrfd) == 0:
            return [frame, None, None, None]
        (face_image_, mat_rev_, roi_box_) = self.fd.get_single_face(crop_size=512, mode="mtcnn_512", apply_roi=True)
        return [
         frame, face_image_, mat_rev_, roi_box_]


class FaceRestoreThread(Linker):

    def __init__(self, queue_list):
        super().__init__(queue_list, fps_counter=True)
        self.gfp = GFPGAN(model_type="GFPGANv1.4", provider="gpu")

    def forward_func(self, something_in):
        src_face_image_ = something_in[1]
        if src_face_image_ is None:
            return [None] + something_in
        face_restore_out_ = self.gfp.forward(src_face_image_)
        torch.cuda.empty_cache()
        return [
         face_restore_out_] + something_in


class FaceParseThread(Linker):

    def __init__(self, queue_list):
        super().__init__(queue_list, fps_counter=True)
        self.face_mask_ = self.get_face_mask(mask_shape=(512, 512))

    def get_face_mask(self, mask_shape):
        mask = np.zeros((512, 512)).astype(np.float32)
        cv2.ellipse(mask, (256, 256), (220, 160), 90, 0, 360, (255, 255, 255), -1)
        thres = 20
        mask[:thres, :] = 0
        mask[-thres:, :] = 0
        mask[:, :thres] = 0
        mask[:, -thres:] = 0
        mask = cv2.stackBlur(mask, (201, 201))
        mask = mask / 255.0
        mask = cv2.resize(mask, mask_shape)
        return mask[(..., np.newaxis)]

    def forward_func(self, something_in):
        if something_in[0] is None:
            return something_in + [None]
        return something_in + [self.face_mask_]


class FaceReverseThread(Linker):

    def __init__(self, queue_list):
        super().__init__(queue_list, fps_counter=True)
        self.counter = 0
        self.start_time = time.time()

    def forward_func(self, something_in):
        face_restore_out = something_in[0]
        src_img_in = something_in[1]
        if face_restore_out is not None:
            mat_rev = something_in[3]
            roi_box = something_in[4]
            face_mask_ = something_in[5]
            restore_roi = CVImage(None).recover_from_reverse_matrix(face_restore_out, (src_img_in[roi_box[1]:roi_box[3],
             roi_box[0]:roi_box[2]]),
              mat_rev,
              img_fg_mask=face_mask_)
            src_img_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = restore_roi
        return [
         src_img_in]


def write_video_chaofen(output_imgs_queue, temp_dir, result_dir, work_id, audio_path, result_queue, width, height, fps, watermark_switch=0, digital_auth=0):
    output_mp4 = os.path.join(temp_dir, "{}-t.mp4".format(work_id))
    fourcc = (cv2.VideoWriter_fourcc)(*"mp4v")
    result_path = os.path.join(result_dir, "{}-r.mp4".format(work_id))
    video_write = cv2.VideoWriter(output_mp4, fourcc, fps, (
     width, height))
    try:
        q0 = Queue(2)
        q1 = Queue(2)
        q2 = Queue(2)
        q3 = Queue(2)
        q4 = Queue(2)
        fdt = FaceDetectThread([q0, q1])
        frt = FaceRestoreThread([q1, q2])
        fpt = FaceParseThread([q2, q3])
        fret = FaceReverseThread([q3, q4])
        cvvwt = CVVideoWriterThread(video_write, [q4])
        threads_list = [
         fdt,frt,fpt,fret,cvvwt]
        for thread_ in threads_list:
            thread_.start()

        while True:
            (state, reason, value_) = output_imgs_queue.get()
            if type(state) == bool:
                if state == True:
                    logger.info("[{}]视频帧队列处理已结束".format(work_id))
                    q0.put(None)
                    for thread_ in threads_list:
                        thread_.join()

                    break
                if type(state) == bool:
                    if state == False:
                        logger.error("[{}]任务视频帧队列 -> 异常原因:[{}]".format(work_id, reason))
                        q0.put(None)
                        for thread_ in threads_list:
                            thread_.join()

                        raise CustomError(reason)
                for result_img in value_:
                    q0.put(result_img)

        video_write.release()
        if watermark_switch == 1 and digital_auth == 1:
            logger.info("[{}]任务需要水印和数字人标识".format(work_id))
            if width > height:
                command = 'ffmpeg -y -i {} -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10,overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(audio_path, output_mp4, GlobalConfig.instance().watermark_path, GlobalConfig.instance().digital_auth_path, result_path)
                logger.info("command:{}".format(command))
            else:
                command = 'ffmpeg -y -i {} -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10,overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(audio_path, output_mp4, GlobalConfig.instance().watermark_path, GlobalConfig.instance().digital_auth_path, result_path)
                logger.info("command:{}".format(command))
        elif watermark_switch == 1 and digital_auth == 0:
            logger.info("[{}]任务需要水印".format(work_id))
            command = 'ffmpeg -y -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10" -c:a aac -crf 15 -strict -2 {}'.format(audio_path, output_mp4, GlobalConfig.instance().watermark_path, result_path)
            logger.info("command:{}".format(command))
        elif watermark_switch == 0 and digital_auth == 1:
            logger.info("[{}]任务需要数字人标识".format(work_id))
            if width > height:
                command = 'ffmpeg -y -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(audio_path, output_mp4, GlobalConfig.instance().digital_auth_path, result_path)
                logger.info("command:{}".format(command))
            else:
                command = 'ffmpeg -y -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(audio_path, output_mp4, GlobalConfig.instance().digital_auth_path, result_path)
                logger.info("command:{}".format(command))
        else:
            command = "ffmpeg -y -i {} -i {} -c:a aac -c:v libx264 -crf 15 -strict -2 {}".format(audio_path, output_mp4, result_path)
            logger.info("command:{}".format(command))
        subprocess.call(command, shell=True)
        print("###### write over")
        result_queue.put([True, result_path])
    except Exception as e:
        try:
            logger.error("[{}]视频帧队列处理异常结束，异常原因:[{}]".format(work_id, e.__str__()))
            result_queue.put([False, "[{}]视频帧队列处理异常结束，异常原因:[{}]".format(work_id, e.__str__())])
        finally:
            e = None
            del e

    else:
        logger.info("后处理进程结束")


def video_synthesis(output_imgs_queue):
    img_id = 0
    st = time.time()
    while True:
        if not output_imgs_queue.empty():
            et = time.time()
            print("表情迁移首次出现耗时======================:", et - st)
            output_imgs = output_imgs_queue.get()
            for img in output_imgs:
                time.sleep(0.03125)
                cv2.imshow("output_imgs", img)
                cv2.waitKey(1)

            st = time.time()
            continue


def hy_fun(wenet_model, audio_path, drivered_path, output_dir, work_id):
    drivered_queue = multiprocessing.Queue(10)
    output_imgs_queue = multiprocessing.Queue(10)
    result_queue = multiprocessing.Queue(1)
    process_list = []
    audio_wenet_feature = get_aud_feat1(audio_path, fps=30, wenet_model=wenet_model)
    process_list.append(Process(target=drivered_video, args=(drivered_queue, drivered_path, audio_wenet_feature)))
    process_list.append(Process(target=audio_transfer, args=(
     drivered_queue, output_imgs_queue)))
    process_list.append(Process(target=write_video, args=(
     output_imgs_queue, output_dir, output_dir, work_id, audio_path, result_queue)))
    [p.start() for p in process_list]
    [p.join() for p in process_list]
    print("主进程结束")
    try:
        result_path = result_queue.get(True, timeout=10)
        return (0, result_path)
    except Empty:
        return (1, 'generate error')


class Status(Enum):
    run = 1
    success = 2
    error = 3

def init_wh_process(in_queue, out_queue):  # [cite: 1920, 1921, 1928]
    face_detector = FaceDetect(cpu=False, model_path='face_detect_utils/resources/')  # [cite: 1950, 1951, 1952]
    plfd = pfpld(cpu=False, model_path='face_detect_utils/resources/')  # [cite: 1952, 1953, 1954]
    logger.info(">>> init_wh_process进程启动")  # [cite: 1955, 1956]

    while True:  # Implied by jump [cite: 2077, 2094]
        try:  # [cite: 1957]
            code, driver_path = in_queue.get()  # [cite: 1957, 1958, 1959, 1960]
            s = time.time()  # [cite: 1960, 1961, 1962]
            wh_list = []  # [cite: 1962]
            cap = cv2.VideoCapture(driver_path)  # [cite: 1963, 1964, 1965]
            count = 0  # [cite: 1965, 1966]
            has_multi_face = False  # [cite: 1966, 1967]

            try:  # [cite: 1967]
                while cap.isOpened() and count < 100:  # [cite: 1968, 1969, 1970, 1971]
                    ret, frame = cap.read()  # [cite: 1971, 1972, 1973]
                    if not ret: break  # [cite: 1974]

                    bboxes = []  # [cite: 1975]
                    kpss = []  # Placeholder, kpss is UNPACK_SEQUENCE target [cite: 1979]
                    try:  # [cite: 1976]
                        bboxes, kpss = face_detector.get_bboxes(frame)  # [cite: 1976, 1977, 1978, 1979]
                    except Exception as e_bbox:  # [cite: 1980, 1981]
                        logger.error(f"[{code}]init_wh exception: {e_bbox}")  # [cite: 1984, 1985, 1986]

                    bboxes_len = len(bboxes)  # [cite: 1990, 1991, 1992]
                    if bboxes_len > 0:  # [cite: 1992, 1993]
                        if bboxes_len > 1:  # [cite: 1994, 1995]
                            has_multi_face = True  # [cite: 1995, 1996]

                        bbox = bboxes[0]  # [cite: 1996, 1997, 1998]
                        x1_cell, y1_cell, x2, y2, score = bbox.astype(int)  # [cite: 1998, 1999, 2000, 2001, 2002]

                        # Bbox adjustment logic
                        x1 = max(x1_cell - int((x2 - x1_cell) * 0.1), 0)
                        x2 = x2 + int((x2 - x1_cell) * 0.1)
                        y2_adjusted = y2 + int((y2 - y1_cell) * 0.1)  # Renamed to avoid conflict
                        y1 = max(y1_cell, 0)

                        face_img = frame[y1:y2_adjusted, x1:x2]  # [cite: 2019, 2020, 2021, 2022]
                        pots = plfd.forward(face_img)[0]  # [cite: 2023, 2024, 2025]

                        # This list comprehension uses x1, y1 as free vars if they were from cell
                        # Assuming x1_cell, y1_cell are the correct ones from the bbox
                        landmarks = np.array([[x1_cell + x, y1_cell + y] for x, y in pots.astype(np.int32)])  #

                        _xmin, _ymin, w, h = cv2.boundingRect(
                            np.array(landmarks))  # [cite: 2033, 2034, 2035, 2036, 2037, 2038]
                        wh_list.append(w / h)  # [cite: 2038, 2039, 2040, 2041]
                    count += 1  # [cite: 2041, 2042, 2043]
            except Exception as e1_loop:  # [cite: 2044, 2045]
                logger.error(f"[{code}]init_wh exception: {e1_loop}")  # [cite: 2048, 2049, 2050]
            finally:  # [cite: 2055]
                cap.release()  # [cite: 2055, 2056, 2057]

            wh_val = 0  # [cite: 2060, 2061]
            if len(wh_list) > 0:  # [cite: 2058, 2059] (Implicit, original compares to 0)
                wh_val = np.mean(np.array(wh_list))  # [cite: 2061, 2062, 2063, 2064, 2065]

            logger.info(
                f"[{code}]init_wh result :[{wh_val}]， cost: {time.time() - s} s")  # [cite: 2065, 2066, 2067, 2068, 2069, 2070]
            torch.cuda.empty_cache()  # [cite: 2070, 2071, 2072]
            out_queue.put([code, wh_val, has_multi_face])  # [cite: 2072, 2073, 2074, 2075]

        except Exception as e_main:  # [cite: 2077, 2078]
            print(traceback.format_exc())  # [cite: 2080, 2081, 2082]
            # The code tries to get e_main.args to format, if e_main has no args, this might be an issue.
            # Safely converting to string.
            out_queue.put([f"init_wh，失败原因:[{str(e_main)}]", "",
                           False])  # Structure: code (as reason), wh, has_multi_face [cite: 2083, 2084, 2085, 2086, 2087, 2088]
            torch.cuda.empty_cache()  # [cite: 2089, 2090]


def init_wh(code, drivered_path):
    s = time.time()
    face_detector = FaceDetect(cpu=False, model_path="face_detect_utils/resources/")
    plfd = pfpld(cpu=False, model_path="face_detect_utils/resources/")
    wh_list = []
    cap = cv2.VideoCapture(drivered_path)
    count = 0
    try:
        try:
            if cap.isOpened():
                while count < 100:
                    (ret, frame) = cap.read()
                    if not ret:
                        break
                    else:
                        try:
                            (bboxes, kpss) = face_detector.get_bboxes(frame)
                        except Exception as e:
                            try:
                                logger.error("[%s]init_wh exception: %s", code, e)
                            finally:
                                e = None
                                del e

                        else:
                            if len(bboxes) > 0:
                                bbox = bboxes[0]
                                (x1, y1, x2, y2, score) = bbox.astype(int)
                                x1 = max(x1 - int((x2 - x1) * 0.1), 0)
                                x2 = x2 + int((x2 - x1) * 0.1)
                                y2 = y2 + int((y2 - y1) * 0.1)
                                y1 = max(y1, 0)
                                face_img = frame[y1:y2, x1:x2]
                                pots = plfd.forward(face_img)[0]
                                landmarks = np.array([[x1 + x, y1 + y] for x, y in pots.astype(np.int32)])
                                (xmin, ymin, w, h) = cv2.boundingRect(np.array(landmarks))
                                wh_list.append(w / h)
                            count += 1

        except Exception as e1:
            try:
                logger.error("[%s]init_wh exception: %s", code, e1)
            finally:
                e1 = None
                del e1

    finally:
        cap.release()

    if len(wh_list) == 0:
        wh = 0
    else:
        wh = np.mean(np.array(wh_list))
    logger.info("[%s]init_wh result :[%s]， cost: %s s", code, wh, time.time() - s)
    torch.cuda.empty_cache()
    return wh


def get_video_info(video_file):
    cap = cv2.VideoCapture(video_file)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    return (
     fps, width, height, fourcc)


def format_video_audio(code, video_path, audio_path, fourcc):
    if fourcc == cv2.VideoWriter_fourcc("H", "2", "6", "4") or fourcc == cv2.VideoWriter_fourcc("a", "v", "c", "1") or fourcc == cv2.VideoWriter_fourcc("h", "2", "6", "4"):
        ffmpeg_command = "ffmpeg -loglevel warning -i %s -crf 15 -vcodec copy -an -y %s"
    else:
        ffmpeg_command = "ffmpeg -loglevel warning -i %s -c:v libx264 -crf 15 -an -y %s"
    video_format = os.path.join(GlobalConfig.instance().temp_dir, code + "_format.mp4")
    ffmpeg_command = ffmpeg_command % (video_path, video_format)
    logger.info("[%s] -> ffmpeg video: %s", code, ffmpeg_command)
    os.system(ffmpeg_command)
    if not os.path.exists(video_format):
        raise Exception("format video error")
    ffmpeg_command = "ffmpeg -loglevel warning -i %s -ac 1 -ar 16000 -acodec pcm_s16le -y  %s"
    audio_format = os.path.join(GlobalConfig.instance().temp_dir, code + "_format.wav")
    ffmpeg_command = ffmpeg_command % (audio_path, audio_format)
    logger.info("[%s] -> ffmpeg audio: %s", code, ffmpeg_command)
    os.system(ffmpeg_command)
    if not os.path.exists(audio_format):
        raise Exception("format audio error")
    return (video_format, audio_format)


def get_license():
    logger.info("license check start ...")
    while True:
        if not check_lc():
            logger.info("license check failed")
        time.sleep(30)


def a():
    if GlobalConfig.instance().register_enable == 1:
        result = register_host()
        if not result:
            raise Exception("服务注册失败.")
        threading.Thread(target=repost_host).start()
    else:
        logger.warning(" -> 服务不进行注册")


class TransDhTask(object):

    def __init__(self, *args, **kwargs):
        logger.info("TransDhTask init")
        set_start_method("spawn", force=True)
        self.run_lock = threading.Lock()
        self.task_dic = {}
        self.run_flag = False
        self.batch_size = int(GlobalConfig.instance().batch_size)
        self.drivered_queue = multiprocessing.Queue(10)
        self.output_imgs_queue = multiprocessing.Queue(10)
        self.result_queue = multiprocessing.Queue(1)
        self.wenet_model = load_ppg_model("wenet/examples/aishell/aidata/conf/train_conformer_multi_cn.yaml", "wenet/examples/aishell/aidata/exp/conformer/wenetmodel.pt", "cuda")
        multiprocessing.Process(target=audio_transfer, args=(
         self.drivered_queue, self.output_imgs_queue, self.batch_size),
          daemon=True).start()
        self.init_wh_queue = multiprocessing.Queue(2)
        self.init_wh_queue_output = multiprocessing.Queue(2)
        multiprocessing.Process(target=init_wh_process, args=(
         self.init_wh_queue, self.init_wh_queue_output),
          daemon=True).start()

    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(TransDhTask, "_instance"):
            TransDhTask._instance = TransDhTask(*args, **kwargs)
        return TransDhTask._instance

    def work(self, audio_url, video_url, code, watermark_switch, digital_auth, chaofen, pn):
        logger.info("任务:{} -> audio_url:{}  video_url:{}".format(code, audio_url, video_url))
        st = time.time()
        self.run_flag = True
        try:
            try:
                self.change_task_status(code, Status.run, 0, "", "")
                try:
                    s1 = time.time()
                    (fps, width, height, fourcc) = get_video_info(video_url)
                    (_tmp_audio_path, _tmp_video_path) = self.preprocess(audio_url, video_url, code)
                    (_video_url, _audio_url) = format_video_audio(code, _tmp_video_path, _tmp_audio_path, fourcc)
                    logger.info("[%s] -> 预处理耗时:%ss", code, time.time() - s1)
                except Exception as e:
                    try:
                        traceback.print_exc()
                        logger.error("[{}]预处理失败，异常信息:[{}]".format(code, e.__str__()))
                        raise CustomError("[{}]预处理失败，异常信息:[{}]".format(code, e.__str__()))
                    finally:
                        e = None
                        del e

                else:
                    if not (os.path.exists(_video_url) and os.path.exists(_audio_url)):
                        raise Exception("视频入参或音频入参下载处理异常")
                    self.change_task_status(code, Status.run, 10, "", "文件下载完成")
                    self.init_wh_queue.put([code, _video_url])
                    try:
                        print(">>> 777   {}".format(fps))
                        s = time.time()
                        audio_wenet_feature = get_aud_feat1(_audio_url, fps=fps, wenet_model=(self.wenet_model))
                        logger.info("[%s] -> get_aud_feat1 cost:%ss", code, time.time() - s)
                    except Exception as e:
                        try:
                            traceback.print_exc()
                            logger.error("[{}]音频特征提取失败，异常信息:[{}]".format(code, e.__str__()))
                            raise CustomError("[{}]音频特征提取失败，异常信息:[{}]".format(code, e.__str__()))
                        finally:
                            e = None
                            del e

                    else:
                        self.change_task_status(code, Status.run, 20, "", "音频特征提取完成")
                        process_list = []
                        wh = 0
                        try:
                            wh_output = self.init_wh_queue_output.get(timeout=10)
                            if wh_output[0] == code:
                                wh = wh_output[1]
                            if wh_output[2]:
                                raise Exception("存在多人脸")
                        except Exception as e1:
                            try:
                                print(traceback.format_exc())
                                raise Exception(e1)
                            finally:
                                e1 = None
                                del e1

                        else:
                            logger.info("[%s] -> wh: [%s]", code, wh)
                            if pn == 1:
                                process_list.append(Process(target=drivered_video_pn, args=(
                                 code, self.drivered_queue, _tmp_video_path, audio_wenet_feature,
                                 self.batch_size, wh),
                                  daemon=True))
                            else:
                                process_list.append(Process(target=drivered_video, args=(
                                 code, self.drivered_queue, _tmp_video_path, audio_wenet_feature,
                                 self.batch_size, wh),
                                  daemon=True))
                            if chaofen == 1 and GlobalConfig.instance().chaofen_after == "1":
                                process_list.append(Process(target=write_video_chaofen, args=(
                                 self.output_imgs_queue, GlobalConfig.instance().temp_dir,
                                 GlobalConfig.instance().result_dir, code,
                                 _tmp_audio_path,
                                 self.result_queue, width, height, fps, watermark_switch, digital_auth),
                                  daemon=True))
                            else:
                                process_list.append(Process(target=write_video, args=(
                                 self.output_imgs_queue, GlobalConfig.instance().temp_dir,
                                 GlobalConfig.instance().result_dir, code,
                                 _tmp_audio_path,
                                 self.result_queue, width, height, fps, watermark_switch, digital_auth),
                                  daemon=True))
                            [p.start() for p in process_list]
                            [p.join() for p in process_list]
                            try:
                                try:
                                    (state, result_path) = self.result_queue.get(True, timeout=10)
                                    print(">>>>>>>>>>>>>>1111 {} {}".format(state, result_path))
                                    if state:
                                        self.change_task_status(code, Status.run, 90, result_path, "视频处理完成")
                                        _remote_file = os.path.join(GlobalConfig.instance().result_dir, "{}.mp4".format(code))
                                        _final_url = result_path
                                        logger.info("[{}]任务最终合成结果: {}".format(code, _final_url))
                                        self.change_task_status(code, Status.success, 100, _final_url, "任务完成")
                                        sweep([GlobalConfig.instance().result_dir], True if GlobalConfig.instance().result_clean_switch == "1" else False)
                                    else:
                                        self.change_task_status(code, Status.error, 0, "", result_path)
                                except Empty:
                                    self.change_task_status(code, Status.error, 0, "", "**生成视频失败")

                            finally:
                                del audio_wenet_feature
                                gc.collect()

            except Exception as e:
                try:
                    traceback.print_exc()
                    logger.error("[{}]任务执行失败，异常信息:[{}]".format(code, e.__str__()))
                    self.change_task_status(code, Status.error, 0, "", e.__str__())
                finally:
                    e = None
                    del e

        finally:
            sweep([GlobalConfig.instance().temp_dir], True if GlobalConfig.instance().temp_clean_switch == "1" else False)
            self.drivered_queue.empty()
            self.output_imgs_queue.empty()
            self.result_queue.empty()
            torch.cuda.empty_cache()
            self.run_flag = False
            logger.info(">>> 任务:{} 耗时:{} ".format(code, time.time() - st))

    def preprocess(self, audio_url, video_url, code):
        s_pre = time.time()
        try:
            if audio_url.startswith("http:") or audio_url.startswith("https:"):
                _tmp_audio_path = os.path.join(GlobalConfig.instance().temp_dir, "{}.wav".format(code))
                download_file(audio_url, _tmp_audio_path)
            else:
                _tmp_audio_path = audio_url
        except Exception as e:
            try:
                traceback.print_exc()
                raise CustomError("[{}]音频下载失败，异常信息:[{}]".format(code, e.__str__()))
            finally:
                e = None
                del e

        else:
            try:
                if video_url.startswith("http:") or video_url.startswith("https:"):
                    _tmp_video_path = os.path.join(GlobalConfig.instance().temp_dir, "{}.mp4".format(code))
                    download_file(video_url, _tmp_video_path)
                else:
                    _tmp_video_path = video_url
            except Exception as e:
                try:
                    traceback.print_exc()
                    raise CustomError("[{}]视频下载失败，异常信息:[{}]".format(code, e.__str__()))
                finally:
                    e = None
                    del e

            else:
                print("--------------------> download cost:", time.time() - s_pre)
                return (
                 _tmp_audio_path, _tmp_video_path)

    def change_task_status(self, code, status: Status, progress: int, result: str, msg=''):
        try:
            try:
                self.run_lock.acquire()
                if code in self.task_dic:
                    self.task_dic[code] = (
                     status, progress, result, msg)
            except Exception as e:
                try:
                    traceback.print_exc()
                    logger.error("[{}]修改任务状态异常，异常信息:[{}]".format(code, e.__str__()))
                finally:
                    e = None
                    del e

        finally:
            self.run_lock.release()


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    wenet_model = load_ppg_model("wenet/examples/aishell/aidata/conf/train_conformer_multi_cn.yaml", "wenet/examples/aishell/aidata/exp/conformer/wenetmodel.pt", "cuda")
    st = time.time()
    result = hy_fun(wenet_model, "test_data/audio/driver_add_valume.wav", "./landmark2face_wy/checkpoints/hy/1.mp4", "./result", 1001)
    print(result, time.time() - st)
