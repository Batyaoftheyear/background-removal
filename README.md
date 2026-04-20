# Background Removal in Real Time

Проект по удалению и замене фона в видео с веб-камеры в реальном времени. Программа работает локально на CPU, использует OpenCV для захвата видео и MediaPipe для сегментации человека.

Обучение модели не выполняется. Используется готовое решение.

## Выбранное решение

Используется библиотека **MediaPipe Selfie Segmentation**.

Ссылка: https://developers.google.com/mediapipe/solutions/vision/image_segmenter

В коде используется:

```python
SelfieSegmentation(model_selection=1)
```

`model_selection=1` - landscape-вариант модели. Он выбран потому, что подходит для горизонтального видео с веб-камеры и достаточно быстро работает на CPU при разрешении 640x480.

Плюсы этого решения для задания:

- модель уже обучена;
- не нужен PyTorch/TensorFlow training pipeline;
- можно запускать локально;
- подходит для real-time обработки;
- легко совмещается с OpenCV.

## Описание архитектуры

MediaPipe Selfie Segmentation - легкая нейросетевая модель для сегментации человека. По смыслу это CNN-based модель: она получает RGB-кадр и возвращает маску переднего плана.

Пайплайн работы программы:

1. OpenCV открывает веб-камеру.
2. Запрашивается разрешение 640x480.
3. Кадр переводится из BGR в RGB.
4. MediaPipe строит маску человека.
5. К маске применяется порог, затем она слегка размывается для более мягких краев.
6. Человек совмещается с выбранным фоном.
7. После сегментации и замены фона считается FPS.
8. Результат показывается в окне OpenCV.

Режимы фона:

- `1` - однотонный фон;
- `2` - размытие исходного фона;
- `3` - фон из `assets/bg.jpg`.

Если файла `assets/bg.jpg` нет, программа не падает и использует цветной фон.

## Запуск

Проверенный вариант - Python 3.11. На Python 3.14 нужный старый модуль MediaPipe Selfie Segmentation может отсутствовать.

Установка зависимостей:

```powershell
cd "c:\Users\anton\VK_mag\background removal"
& "C:\Users\anton\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Запуск:

```powershell
python app.py
```

Если виртуальное окружение не активировано:

```powershell
.\.venv\Scripts\python.exe app.py
```

Управление:

| Клавиша | Действие |
| --- | --- |
| `1` | однотонный фон |
| `2` | размытие фона |
| `3` | фон из `assets/bg.jpg` |
| `q` | выход |
| `Esc` | выход |

Основные параметры в `app.py`:

```python
CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
MASK_THRESHOLD = 0.55
MASK_BLUR = (7, 7)
BG_IMAGE_PATH = os.path.join("assets", "bg.jpg")
```

## Результаты

- Разрешение: `640x480`
- FPS по обработанным кадрам: `49.6`
- Устройство: `AMD Ryzen 5 4500U with Radeon Graphics`
- Качество: `человек выделяется стабильно, фон заменяется в реальном времени. Иногда появляются артефакты по краям силуэта и рядом с плечами, особенно если на фоне есть яркие объекты.`

## Демо

Ссылка на видео: https://disk.yandex.ru/i/HSM4yrK-yLvsNA
