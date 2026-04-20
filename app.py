import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from mediapipe.python.solutions import selfie_segmentation
except ImportError as err:
    selfie_segmentation = None
    MEDIAPIPE_ERROR = err


WINDOW_NAME = "Background removal"
CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480

MASK_THRESHOLD = 0.55
MASK_BLUR = (7, 7)

COLOR_BG = (60, 170, 230)  # BGR
BG_IMAGE_PATH = os.path.join("assets", "bg.jpg")


@dataclass
class FpsCounter:
    frames: int = 0
    total_time: float = 0.0
    current: float = 0.0
    average: float = 0.0

    def update(self, frame_time: float) -> None:
        self.frames += 1
        self.total_time += frame_time
        self.current = 1.0 / frame_time if frame_time > 0 else 0.0
        self.average = self.frames / self.total_time if self.total_time > 0 else 0.0


def check_mediapipe() -> None:
    if selfie_segmentation is not None:
        return

    raise RuntimeError(
        "Не найден MediaPipe Selfie Segmentation.\n"
        "Для этого проекта нужен Python 3.11 и mediapipe==0.10.14.\n"
        "Создайте окружение заново и установите зависимости из requirements.txt.\n"
        f"Ошибка импорта: {MEDIAPIPE_ERROR}"
    )


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру. Возможно, она занята другой программой.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    real_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Запрошено разрешение: {WIDTH}x{HEIGHT}")
    print(f"Камера отдала: {real_width}x{real_height}")

    return cap


def make_color_background() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), COLOR_BG, dtype=np.uint8)


def load_background_image() -> np.ndarray | None:
    if not os.path.exists(BG_IMAGE_PATH):
        print(f"Файл {BG_IMAGE_PATH} не найден. Для режима 3 будет использован цветной фон.")
        return None

    image = cv2.imread(BG_IMAGE_PATH)
    if image is None:
        print(f"Не удалось прочитать {BG_IMAGE_PATH}. Для режима 3 будет использован цветной фон.")
        return None

    return cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def make_alpha_mask(mask: np.ndarray) -> np.ndarray:
    # Порог убирает слабые ложные срабатывания, размытие смягчает край силуэта.
    alpha = (mask > MASK_THRESHOLD).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, MASK_BLUR, 0)
    return np.clip(alpha, 0.0, 1.0)


def compose(frame: np.ndarray, alpha: np.ndarray, mode: int, color_bg: np.ndarray, image_bg: np.ndarray | None) -> np.ndarray:
    if mode == 2:
        background = cv2.GaussianBlur(frame, (55, 55), 0)
    elif mode == 3 and image_bg is not None:
        background = image_bg
    else:
        background = color_bg

    alpha = alpha[:, :, None]
    result = frame.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha)
    return cv2.convertScaleAbs(result)


def draw_info(frame: np.ndarray, mode: int, fps: FpsCounter) -> np.ndarray:
    mode_names = {
        1: "color",
        2: "blur",
        3: "image",
    }

    lines = [
        f"Mode: {mode} ({mode_names[mode]})",
        f"FPS: {fps.current:.1f}",
        f"Avg FPS: {fps.average:.1f}",
        "1 color | 2 blur | 3 image | q/Esc exit",
    ]

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 118), (0, 0, 0), -1)

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 28 + i * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def main() -> None:
    cap = None

    try:
        check_mediapipe()

        cap = open_camera()
        color_bg = make_color_background()
        image_bg = load_background_image()

        mode = 1
        fps = FpsCounter()

        with selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Кадр с камеры не прочитан.")
                    break

                frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                start = time.perf_counter()

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = segmenter.process(rgb)
                rgb.flags.writeable = True

                alpha = make_alpha_mask(result.segmentation_mask)
                output = compose(frame, alpha, mode, color_bg, image_bg)

                fps.update(time.perf_counter() - start)
                output = draw_info(output, mode, fps)

                cv2.imshow(WINDOW_NAME, output)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("1"):
                    mode = 1
                elif key == ord("2"):
                    mode = 2
                elif key == ord("3"):
                    mode = 3

    except RuntimeError as err:
        print(f"Ошибка: {err}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
