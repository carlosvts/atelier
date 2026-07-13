from dataclasses import dataclass
from pathlib import Path
import math
import time

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


CAMERA_INDEX = 0
MODEL_PATH = Path("models/hand_landmarker.task")


# Índices fixos dos landmarks da mão no MediaPipe.
# Esses índices sempre representam a mesma parte anatômica da mão.
WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


# Conexões entre os landmarks.
# O MediaPipe nos dá pontos soltos; essas conexões são nossas,
# só para desenhar o "esqueleto" visual da mão.
HAND_CONNECTIONS = [
    # Polegar
    (WRIST, THUMB_CMC),
    (THUMB_CMC, THUMB_MCP),
    (THUMB_MCP, THUMB_IP),
    (THUMB_IP, THUMB_TIP),

    # Indicador
    (WRIST, INDEX_MCP),
    (INDEX_MCP, INDEX_PIP),
    (INDEX_PIP, INDEX_DIP),
    (INDEX_DIP, INDEX_TIP),

    # Médio
    (WRIST, MIDDLE_MCP),
    (MIDDLE_MCP, MIDDLE_PIP),
    (MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_DIP, MIDDLE_TIP),

    # Anelar
    (WRIST, RING_MCP),
    (RING_MCP, RING_PIP),
    (RING_PIP, RING_DIP),
    (RING_DIP, RING_TIP),

    # Mindinho
    (WRIST, PINKY_MCP),
    (PINKY_MCP, PINKY_PIP),
    (PINKY_PIP, PINKY_DIP),
    (PINKY_DIP, PINKY_TIP),

    # Base da palma
    (INDEX_MCP, MIDDLE_MCP),
    (MIDDLE_MCP, RING_MCP),
    (RING_MCP, PINKY_MCP),
]


@dataclass
class PinchInfo:
    """
    Guarda os dados principais da pinça.

    thumb_tip:
        posição da ponta do polegar em pixels.

    index_tip:
        posição da ponta do indicador em pixels.

    center:
        ponto médio entre polegar e indicador.
        Esse será nosso "cursor virtual".

    distance_px:
        distância em pixels entre polegar e indicador.

    hand_scale_px:
        tamanho aproximado da mão em pixels.

    ratio:
        distância da pinça normalizada pelo tamanho da mão.
        Isso é melhor do que usar apenas pixels, porque a mão pode estar
        mais perto ou mais longe da câmera.
    """

    thumb_tip: tuple[int, int]
    index_tip: tuple[int, int]
    center: tuple[int, int]
    distance_px: float
    hand_scale_px: float
    ratio: float


class RectangleDrawer:
    """
    Pequena máquina de estado para desenhar retângulos usando pinça.

    Estados principais:
    - sem pinça: nada está sendo desenhado
    - pinça começou: salva o primeiro canto do retângulo
    - pinça continua: atualiza o segundo canto
    - pinça soltou: fixa o retângulo na tela
    """

    def __init__(
        self,
        close_threshold: float = 0.35,
        open_threshold: float = 0.50,
        min_rect_size_px: int = 12,
    ):
        # Threshold menor para começar pinça.
        self.close_threshold = close_threshold

        # Threshold maior para soltar pinça.
        # Essa diferença cria "histerese" e evita flickering:
        # se a distância oscilar perto do limite, o estado não fica
        # alternando loucamente entre fechado/aberto.
        self.open_threshold = open_threshold

        self.min_rect_size_px = min_rect_size_px

        self.is_pinching = False
        self.start_point: tuple[int, int] | None = None
        self.current_point: tuple[int, int] | None = None

        # Lista de retângulos finalizados.
        # Cada retângulo é salvo como: (ponto_inicial, ponto_final)
        self.finished_rectangles: list[
            tuple[tuple[int, int], tuple[int, int]]
        ] = []

    def update(self, pinch_info: PinchInfo):
        """
        Atualiza o estado do retângulo com base na pinça atual.
        """

        was_pinching = self.is_pinching

        # Se já estava pinçando, só solta quando passar do open_threshold.
        if self.is_pinching:
            self.is_pinching = pinch_info.ratio < self.open_threshold

        # Se não estava pinçando, só começa quando passar do close_threshold.
        else:
            self.is_pinching = pinch_info.ratio < self.close_threshold

        # Transição: abriu -> fechou.
        # momento em que o usuário começou a desenhar.
        if self.is_pinching and not was_pinching:
            self.start_point = pinch_info.center
            self.current_point = pinch_info.center

        # Estado contínuo: continua fechado.
        # Aqui o usuário está arrastando o segundo canto do retângulo.
        elif self.is_pinching and was_pinching:
            self.current_point = pinch_info.center

        # Transição: fechou -> abriu.
        # Esse é o momento em que o retângulo é finalizado.
        elif not self.is_pinching and was_pinching:
            self._finish_current_rectangle()

    def cancel_current_gesture(self):
        """
        Cancela o gesto atual.

        Isso é útil quando a mão some da câmera.
        Nesse caso, é melhor cancelar o retângulo em andamento
        do que finalizar um retângulo acidental.
        """

        self.is_pinching = False
        self.start_point = None
        self.current_point = None

    def clear(self):
        """
        Apaga todos os retângulos já desenhados.
        """

        self.finished_rectangles.clear()
        self.cancel_current_gesture()

    def _finish_current_rectangle(self):
        if self.start_point is None or self.current_point is None:
            self.start_point = None
            self.current_point = None
            return

        x1, y1 = self.start_point
        x2, y2 = self.current_point

        width = abs(x2 - x1)
        height = abs(y2 - y1)

        # Evita salvar retângulos minúsculos quando o usuário só fecha
        # e abre a pinça sem arrastar de verdade.
        if width >= self.min_rect_size_px and height >= self.min_rect_size_px:
            self.finished_rectangles.append(
                (self.start_point, self.current_point)
            )

        self.start_point = None
        self.current_point = None

    def draw(self, frame):
        """
        Desenha os retângulos finalizados e o retângulo atual.
        """

        # Retângulos já confirmados.
        for start_point, end_point in self.finished_rectangles:
            cv.rectangle(
                frame,
                start_point,
                end_point,
                (255, 255, 255),
                2,
            )

        # Retângulo que ainda está sendo arrastado.
        if self.start_point is not None and self.current_point is not None:
            cv.rectangle(
                frame,
                self.start_point,
                self.current_point,
                (0, 255, 0),
                2,
            )


def create_hand_landmarker(model_path: Path):
    """
    Cria o modelo de detecção de mãos.
    """

    base_options = python.BaseOptions(
        model_asset_path=str(model_path),
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,

        # VIDEO é o modo certo para processar frames sequenciais.
        # Nesse modo, a gente precisa passar timestamp_ms a cada frame.
        running_mode=vision.RunningMode.VIDEO,

        # Detecta até 2 mãos.
        num_hands=2,

        # Thresholds de confiança.
        # Pode aumentar se estiver tendo falso positivo.
        # Pode diminuir se a mão não estiver sendo detectada.
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.HandLandmarker.create_from_options(options)


def frame_to_mp_image(frame_bgr):
    """
    OpenCV usa BGR.
    MediaPipe espera RGB.
    precisamos converter antes de mandar para o modelo.
    """

    frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb,
    )


def landmark_to_pixel(landmark, frame_width: int, frame_height: int):
    """
    MediaPipe retorna x e y normalizados entre 0 e 1.

    Exemplo:
    - landmark.x = 0.5 significa metade da largura
    - landmark.y = 0.25 significa 25% da altura

    Aqui transformamos isso em coordenadas reais de pixel.
    """

    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)

    return x, y


def distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Distância Euclidiana 2D entre dois pontos.
    """

    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]

    return math.sqrt(dx * dx + dy * dy)


def midpoint(p1: tuple[int, int], p2: tuple[int, int]):
    """
    Ponto médio entre dois pontos.

    No nosso caso, o ponto médio entre polegar e indicador
    vira o cursor da pinça.
    """

    x = (p1[0] + p2[0]) // 2
    y = (p1[1] + p2[1]) // 2

    return x, y


def compute_pinch_info(
    hand_landmarks,
    frame_width: int,
    frame_height: int,
) -> PinchInfo:
    """
    Extrai as informações da pinça a partir dos landmarks da mão.

    A pinça é definida pela distância entre:
    - landmark 4: ponta do polegar
    - landmark 8: ponta do indicador
    """

    thumb_tip = landmark_to_pixel(
        hand_landmarks[THUMB_TIP],
        frame_width,
        frame_height,
    )

    index_tip = landmark_to_pixel(
        hand_landmarks[INDEX_TIP],
        frame_width,
        frame_height,
    )

    pinch_center = midpoint(thumb_tip, index_tip)
    pinch_distance = distance(thumb_tip, index_tip)

    # Uma escala simples para o tamanho da mão.
    # Usamos a distância entre o pulso e a base do dedo médio.
    # Isso ajuda a normalizar a pinça.
    wrist = landmark_to_pixel(
        hand_landmarks[WRIST],
        frame_width,
        frame_height,
    )

    middle_mcp = landmark_to_pixel(
        hand_landmarks[MIDDLE_MCP],
        frame_width,
        frame_height,
    )

    hand_scale = distance(wrist, middle_mcp)

    if hand_scale <= 1e-6:
        ratio = 1.0
    else:
        ratio = pinch_distance / hand_scale

    return PinchInfo(
        thumb_tip=thumb_tip,
        index_tip=index_tip,
        center=pinch_center,
        distance_px=pinch_distance,
        hand_scale_px=hand_scale,
        ratio=ratio,
    )


def draw_hand_landmarks(frame, hand_landmarks):
    """
    Desenha os 21 pontos da mão e as linhas entre eles.

    MediaPipe só fornece os pontos.
    O desenho é todo feito por OpenCV.
    """

    height, width, _ = frame.shape

    points = [
        landmark_to_pixel(landmark, width, height)
        for landmark in hand_landmarks
    ]

    # Desenha conexões primeiro.
    for start_idx, end_idx in HAND_CONNECTIONS:
        start_point = points[start_idx]
        end_point = points[end_idx]

        cv.line(
            frame,
            start_point,
            end_point,
            (255, 0, 0),
            2,
        )

    # Desenha os pontos por cima.
    for index, point in enumerate(points):
        cv.circle(
            frame,
            point,
            4,
            (0, 255, 0),
            -1,
        )

        # desenha o índice de cada landmark para proposito de estudo.
        cv.putText(
            frame,
            str(index),
            (point[0] + 4, point[1] - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )


def draw_pinch_debug(frame, pinch_info: PinchInfo, is_pinching: bool):
    """
    Desenha informações visuais da pinça.
    """

    color = (0, 255, 0) if is_pinching else (0, 0, 255)

    # Linha entre polegar e indicador.
    cv.line(
        frame,
        pinch_info.thumb_tip,
        pinch_info.index_tip,
        color,
        2,
    )

    # Cursor virtual no centro da pinça.
    cv.circle(
        frame,
        pinch_info.center,
        8,
        color,
        -1,
    )

    text = (
        f"pinch ratio: {pinch_info.ratio:.2f} | "
        f"distance: {pinch_info.distance_px:.1f}px"
    )

    cv.putText(
        frame,
        text,
        (20, 60),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
    )

    state_text = "PINCHING" if is_pinching else "OPEN"

    cv.putText(
        frame,
        state_text,
        (20, 90),
        cv.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
    )


def draw_handedness(frame, hand_result):
    """
    Desenha se o MediaPipe acha que a mão é esquerda ou direita.
    """

    if not hand_result.handedness:
        return

    y = 120

    for i, handedness in enumerate(hand_result.handedness):
        if not handedness:
            continue

        category = handedness[0]

        text = f"hand {i}: {category.category_name} ({category.score:.2f})"

        cv.putText(
            frame,
            text,
            (20, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
        )

        y += 26


def draw_debug_info(frame, fps: float, hand_count: int):
    """
    Informações gerais do programa.
    """

    cv.putText(
        frame,
        f"FPS: {fps:.1f} | hands: {hand_count}",
        (20, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        1,
    )

    cv.putText(
        frame,
        "q: quit | c: clear rectangles",
        (20, frame.shape[0] - 20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
    )


def camera_loop(camera_index: int):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Download hand_landmarker.task into the models/ directory."
        )

    cap = cv.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    rectangle_drawer = RectangleDrawer()

    start_time = time.monotonic()
    previous_time = time.monotonic()
    fps = 0.0

    try:
        with create_hand_landmarker(MODEL_PATH) as hand_landmarker:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                now = time.monotonic()
                delta = now - previous_time
                previous_time = now

                if delta > 0:
                    fps = 1.0 / delta

                # Timestamp em milissegundos desde o início do programa.
                # O modo VIDEO do MediaPipe precisa disso.
                timestamp_ms = int((now - start_time) * 1000)

                # Converte frame OpenCV para imagem MediaPipe.
                mp_image = frame_to_mp_image(frame)

                # o modelo recebe a imagem e retorna landmarks da mão.
                hand_result = hand_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                hand_count = (
                    len(hand_result.hand_landmarks)
                    if hand_result.hand_landmarks
                    else 0
                )

                # Por enquanto, vamos usar só a primeira mão detectada
                # para controlar o retângulo.
                if hand_result.hand_landmarks:
                    first_hand_landmarks = hand_result.hand_landmarks[0]

                    height, width, _ = frame.shape

                    pinch_info = compute_pinch_info(
                        first_hand_landmarks,
                        width,
                        height,
                    )

                    # Atualiza a máquina de estado do retângulo.
                    rectangle_drawer.update(pinch_info)

                    # Desenha a mão e a pinça.
                    draw_hand_landmarks(frame, first_hand_landmarks)
                    draw_pinch_debug(
                        frame,
                        pinch_info,
                        rectangle_drawer.is_pinching,
                    )

                else:
                    # Se a mão sumiu, cancelamos o gesto atual para evitar
                    # finalizar retângulo sem querer.
                    rectangle_drawer.cancel_current_gesture()

                # Desenha retângulos depois da lógica da mão.
                rectangle_drawer.draw(frame)

                draw_handedness(frame, hand_result)
                draw_debug_info(frame, fps, hand_count)

                cv.imshow("pinch rectangle drawer", frame)

                key = cv.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("c"):
                    rectangle_drawer.clear()

    finally:
        cap.release()
        cv.destroyAllWindows()


def main():
    camera_loop(CAMERA_INDEX)


if __name__ == "__main__":
    main()