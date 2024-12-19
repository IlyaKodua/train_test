import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Tuple

class VelocityCalculator:
    """
    Класс для расчета скорости движения объектов на видео с помощью оптического потока и фильтра Калмана.
    Позволяет добавлять визуализацию вектора скорости на каждый кадр видео.
    """

    def __init__(self, frame_step : int = 5, kalman_dt : float = 0.1) -> None:
        """
        Конструктор класса.
        
        Параметры:
        - frame_step (int): количество кадров между шагами обработки (по умолчанию 5).
        - kalman_dt (float): временной интервал для фильтра Калмана (по умолчанию 0.1 секунды).
        """
        # Инициализация фильтра Калмана
        self.kf = self.kalman_filter_init_with_acceleration(kalman_dt)
        # Установка шага обработки кадров
        self.frame_step = frame_step

    def step(self, frame : np.ndarray, prvs :np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обработка одного кадра.
        
        Параметры:
        - frame (np.ndarray): текущий кадр видео.
        - prvs (np.ndarray): предыдущий кадр видео.
        
        Возвращает:
        - frame_with_vector (np.ndarray): кадр с добавленным вектором скорости.
        - next_frame (np.ndarray): следующий обработанный кадр.
        """
        next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Вычисление оптического потока между предыдущим и текущим кадром
        flow = cv2.calcOpticalFlowFarneback(
            prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        # Разделение оптического потока на компоненты полярных координат
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Фильтрация оптического потока
        v = self.magfilter(flow, mag)
        self.kf.predict()
        self.kf.update(v)
        # Извлечение текущей оценки скорости
        velocity = self.kf.x[0:2]
        frame_with_vector = self.draw_vector(frame, velocity)
        return frame_with_vector, next_frame

    def get_video_with_velocity(self, input_video_path : str, output_video_path : str) -> None:
        """
        Обработка всего видео и сохранение результата.
        
        Параметры:
        - input_video_path (str): путь к входному видеофайлу.
        - output_video_path (str): путь к выходному видеофайлу.
        """
        cap = cv2.VideoCapture(input_video_path)
        ret, frame = cap.read()
        prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fps = cap.get(cv2.CAP_PROP_FPS)
        result = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"MP4V"),
            fps,
            (int(cap.get(3)), int(cap.get(4))),
        )
        cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if cnt % self.frame_step == 0:
                frame_with_vector, prvs = self.step(frame, prvs)
                result.write(frame_with_vector)
            cnt += 1
        result.release()
        cap.release()

    @staticmethod
    def draw_vector(frame : np.ndarray, velocity : np.ndarray,
                    start_point : Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Рисование вектора скорости на кадре.
        
        Параметры:
        - frame (np.ndarray) : кадр, на который нужно добавить вектор скорости.
        - velocity (np.ndarray): вектор скорости.
        - start_point (Tuple[int, int]): начальная точка для рисования вектора (по умолчанию (100, 100)).
        
        Возвращает:
        - frame (np.ndarray): кадр с нарисованным вектором скорости.
        """
        # Масштабирование вектора скорости для отображения
        v_mean = velocity * 50 / (1e-6 + np.linalg.norm(velocity))
        v_mean = np.array(v_mean).astype(int)
        # Рисование стрелки, представляющей вектор скорости
        cv2.circle(frame, start_point, 50, (255,255,255), -1)
        cv2.circle(frame, start_point, 52, (0,0,0), 2)
        cv2.arrowedLine(
            frame, start_point, np.asarray(start_point) + v_mean, (0, 0, 255), 2, 2
        )
        return frame

    @staticmethod
    def kalman_filter_init_with_acceleration(dt : float) -> KalmanFilter:
        """
        Инициализация фильтра Калмана с учетом ускорения.
        
        Параметр:
        - dt (float): временной интервал для обновления состояния фильтра Калмана.
        
        Возвращает:
        - kf (KalmanFilter): объект фильтра Калмана, настроенный для учета ускорения.
        """
        # Размерность состояния и измерения
        dim_x = 4
        dim_z = 2

        # Создание объекта фильтра Калмана
        kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

        # Матрица перехода состояний
        kf.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # Матрица наблюдения
        kf.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )

        # Ковариационная матрица процесса
        kf.Q *= np.eye(dim_x) * 0.001
        # Ковариационная матрица измерений
        kf.R *= np.eye(dim_z) * 1

        # Начальное состояние
        kf.x = np.zeros(dim_x)
        # Начальная ковариационная матрица ошибок
        kf.P *= np.eye(dim_x)

        return kf

    @staticmethod
    def magfilter(flow : np.ndarray, mag : np.ndarray, thres : float = 0.7) -> np.ndarray:
        """
        Фильтрация оптического потока.
        
        Параметры:
        - flow: массив значений оптического потока.
        - mag: массив величин оптического потока.
        - thres (float): пороговое значение для фильтрации малых движений (по умолчанию 0.7).
        
        Возвращает:
        - v_mean: среднее значение скоростей, превышающих порог.
        """
        # Отбор только тех значений, которые превышают порог
        v = flow[mag >= thres * np.max(mag)]
        # Расчёт среднего значения отобранных скоростей
        v_mean = np.mean(v, axis=0)
        return v_mean