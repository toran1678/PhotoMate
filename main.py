import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QColorDialog, QSlider, QHBoxLayout, QFrame, QWidget, QComboBox, QLineEdit, QMenu, QMessageBox, QAction, QToolTip, QDialog
)
import os
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, QPoint
from urllib.parse import quote

class PhotoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Main Window 설정
        self.setWindowTitle("PhotoMate")
        self.setGeometry(100, 100, 1200, 800)
        #self.setFixedSize(1200, 900)  # 창의 최대 크기 제한

        self.setWindowIcon(QIcon("./icons/photomate.png"))

        # 상단 메뉴 생성
        self.create_menu_bar()

        # 중앙 위젯 설정
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # 메인 레이아웃
        self.main_layout = QHBoxLayout(self.central_widget)

        # 좌측 사이드바 레이아웃
        self.sidebar_layout = QVBoxLayout()
        self.sidebar_layout.setSpacing(5) # 버튼 간 간격
        #self.sidebar_layout.setContentsMargins(5, 5, 5, 5)  # 여백 감소
        #self.sidebar_layout.setAlignment(Qt.AlignTop)  # 상단 정렬

        # 도형 선택 드롭다운 추가
        self.shape_selector = QComboBox(self)
        self.shape_selector.setToolTip("그리기 도구를 선택합니다 (펜, 선, 도형, 텍스트).")
        self.shape_selector.addItem("펜 도구")  # 기본 펜 도구
        self.shape_selector.addItem("선 그리기")
        self.shape_selector.addItem("원 그리기")
        self.shape_selector.addItem("삼각형 그리기")
        self.shape_selector.addItem("사각형 그리기")
        self.shape_selector.addItem("텍스트 그리기")
        self.shape_selector.setFixedHeight(60)  # 높이 설정
        self.shape_selector.currentIndexChanged.connect(self.change_shape)  # 도형 선택 시 이벤트 연결
        self.sidebar_layout.addWidget(self.shape_selector)

        # 텍스트 입력창과 확인 버튼 추가
        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("텍스트 도구 문구")
        self.text_input.setFixedWidth(200)
        self.text_input.hide()  # 기본적으로 숨김
        self.sidebar_layout.addWidget(self.text_input)

        self.text_confirm_button = QPushButton("확인", self)
        self.text_confirm_button.setFixedWidth(200)
        self.text_confirm_button.clicked.connect(self.confirm_text_input)
        self.text_confirm_button.hide()  # 기본적으로 숨김
        self.sidebar_layout.addWidget(self.text_confirm_button)

        # 사이드바 버튼 추가
        self.add_sidebar_button("기본 모드", self.apply_default_mode, tooltip="기본 모드로 변경합니다.")
        self.add_sidebar_button("되돌리기", self.undo, tooltip="이전 상태로 되돌립니다.")
        self.add_sidebar_button("앞으로 가기", self.redo, tooltip="되돌리기를 취소하고 다음 상태로 이동합니다.")
        self.add_sidebar_button("이미지 자르기", self.crop_image, tooltip="이미지의 선택한 영역만 남기고 자릅니다.")

        # self.add_sidebar_button("흑백 변환", self.convert_to_grayscale, tooltip="이미지를 흑백으로 변환합니다.")

        # self.add_sidebar_button("어댑티브 쓰레시홀드", self.apply_adaptive_threshold, tooltip="이미지 밝기에 따라 어댑티브 쓰레시홀드 적용.")
        self.add_sidebar_button("알파 블렌딩", self.apply_alpha_blending, tooltip="이미지를 혼합하여 알파 블렌딩을 적용합니다.")
        self.add_sidebar_button("크로마키 합성", self.apply_chromakey, tooltip="크로마키 배경을 새로운 이미지로 대체합니다.")
        # self.add_sidebar_button("정규화(화질 개선)", self.apply_normalization, tooltip="이미지 밝기 및 대비를 정규화하여 화질을 개선합니다.")
        self.add_sidebar_button("역투영", self.apply_back_projection, tooltip="이미지의 특정 영역을 기반으로 역투영을 수행합니다.")
        self.add_sidebar_button("왜곡(볼록 렌즈)", self.apply_distortion, tooltip="이미지에 볼록 렌즈 효과를 적용합니다.")
        self.add_sidebar_button("방사 왜곡", self.apply_radial_distortion, tooltip="방사 왜곡을 사용하여 이미지의 원근감을 변경합니다.")
        self.add_sidebar_button("모자이크", self.apply_mosaic, tooltip="선택한 영역에 모자이크를 적용합니다.")
        self.add_sidebar_button("블러 모자이크", self.apply_blur_mosaic, tooltip="선택한 영역을 블러 처리하여 모자이크 효과를 적용합니다.")
        self.add_sidebar_button("리퀴파이", self.apply_liquify, tooltip="이미지의 일부를 변형하여 리퀴파이 효과를 적용합니다.")
        # self.add_sidebar_button("블러", self.apply_blur, tooltip="이미지에 블러 효과를 적용합니다.")
        # self.add_sidebar_button("엣지 검출", self.apply_canny_edge_detection, tooltip="이미지에서 엣지를 검출합니다.")
        # self.add_sidebar_button("코너 검출", self.apply_corner_detection, tooltip="이미지의 코너(모서리) 부분을 검출합니다.")
        # self.add_sidebar_button("FAST 특징 검출", self.apply_fast_detection, tooltip="FAST 알고리즘을 사용하여 이미지 특징을 검출합니다.")
        # self.add_sidebar_button("Bolb 검출", self.apply_blob_detection, tooltip="이미지에서 Blob 형태를 검출합니다.")
        # self.add_sidebar_button("열림 연산", self.apply_opening, tooltip="이미지에 열림 연산(침식 후 팽창)을 적용합니다.")
        # self.add_sidebar_button("닫힘 연산", self.apply_closing, tooltip="이미지에 닫힘 연산(팽창 후 침식)을 적용합니다.")
        # self.add_sidebar_button("모폴로지 그레디언트", self.apply_morphological_gradient, tooltip="이미지에 모폴로지 그레디언트를 적용합니다.")
        # self.add_sidebar_button("스케치/페인팅 효과", self.apply_sketch_paint, tooltip="이미지를 스케치 및 페인팅 효과로 변환합니다.")
        self.add_sidebar_button("색상 채우기", self.apply_flood_fill, tooltip="클릭한 영역을 지정된 색상으로 채웁니다.")
        self.add_sidebar_button("워터쉐드(배경 분리)", self.apply_watershed, tooltip="워터쉐드 알고리즘을 사용하여 배경과 객체를 분리합니다.")
        self.add_sidebar_button("그랩컷(배경 분리)", self.apply_grabcut, tooltip="그랩컷 알고리즘으로 전경과 배경을 분리합니다.")
        self.add_sidebar_button("문서 스캔", self.apply_document_scan, tooltip="이미지를 스캔한 문서처럼 보이게 처리합니다.")
        self.add_sidebar_button("템플릿 매칭", self.apply_template_matching, tooltip="이미지에서 템플릿을 검색하여 매칭합니다.")
        self.add_sidebar_button("ORB 매칭", self.apply_orb_matching, tooltip="ORB 알고리즘을 사용하여 이미지 간의 특징을 매칭합니다.")
        self.add_sidebar_button("파노라마 생성", self.create_panorama, tooltip="여러 이미지를 하나로 이어 파노라마를 생성합니다.")


        # 좌측 사이드바 추가
        self.main_layout.addLayout(self.sidebar_layout)

        # 중앙 영역 레이아웃
        self.editor_layout = QVBoxLayout()

        # 상단 버튼 레이아웃
        self.button_layout = QHBoxLayout()

        # 파일 열기 버튼
        # self.open_button = QPushButton("파일 열기", self)
        # self.open_button.clicked.connect(self.open_file)
        # self.button_layout.addWidget(self.open_button)

        # 파일 저장 버튼
        # self.save_button = QPushButton("파일 저장", self)
        # self.save_button.clicked.connect(self.save_file)
        # self.button_layout.addWidget(self.save_button)

        self.brush_color = (0, 0, 0)  # 기본 색상 (검정색)
        # 색상 표시 QLabel 생성
        self.color_display = QLabel(self)
        self.color_display.setFixedSize(30, 30)  # 크기 설정
        self.color_display.setStyleSheet(f"background-color: rgb{self.brush_color}; border: 1px solid black;")
        self.button_layout.addWidget(self.color_display)

        # 색상 선택 버튼
        self.color_button = QPushButton("색상 선택", self)
        self.color_button.clicked.connect(self.choose_color)
        self.button_layout.addWidget(self.color_button)

        # 브러쉬 크기 레이블
        self.brush_size_label = QLabel("브러쉬 크기 선택", self)
        self.button_layout.addWidget(self.brush_size_label)

        # 브러쉬 크기 슬라이더
        self.brush_size_slider = QSlider(Qt.Horizontal, self)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(50)
        self.brush_size_slider.setValue(5)
        self.brush_size_slider.setTickPosition(QSlider.TicksBelow)
        self.brush_size_slider.setTickInterval(1)
        self.button_layout.addWidget(self.brush_size_slider)

        # 이미지 확대/축소 레이블
        self.scale_label = QLabel("이미지 확대/축소", self)
        self.button_layout.addWidget(self.scale_label)

        # 확대/축소 슬라이더
        self.scale_slider = QSlider(Qt.Horizontal, self)
        self.scale_slider.setMinimum(10)
        self.scale_slider.setMaximum(200)
        self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(self.scale_image)
        self.button_layout.addWidget(self.scale_slider)

        # 버튼 레이아웃 추가
        self.editor_layout.addLayout(self.button_layout)

        # 이미지 표시 영역
        self.image_label = QLabel(self)
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.editor_layout.addWidget(self.image_label)

        #self.image_label.setMaximumSize(1100, 900)  # QLabel의 최대 크기 제한

        # 중앙 영역 레이아웃 추가
        self.main_layout.addLayout(self.editor_layout)

        # 기본 값 초기화
        self.image = None  # OpenCV 이미지 (numpy 배열)
        self.temp_image = None  # 그림을 그릴 임시 이미지
        self.scaled_image = None  # 현재 스케일링된 이미지
        self.scale_factor = 1.0  # 이미지 확대/축소 비율
        self.brush_color = (0, 0, 0)  # 기본 색상 (검정색)
        self.drawing = False  # 그림 그리기 상태
        self.last_point = None  # 이전 마우스 위치

        # 초기 모드 설정
        self.current_mode = 'default'

        # 도형 그리기
        self.current_shape = None  # 현재 선택된 도형 (None, 'circle', 'triangle', 'rectangle', 'line')
        self.start_point = None  # 드래그 시작 위치
        self.end_point = None  # 드래그 종료 위치

        # 작업 히스토리 및 앞으로 가기 스택
        self.history = [] # 되돌리기
        self.redo_stack = [] # 앞으로 가기\

    def create_menu_bar(self):
        """상단 메뉴를 생성."""
        menu_bar = self.menuBar()

        # 파일 메뉴
        file_menu = menu_bar.addMenu("파일")
        self.add_action(file_menu, "새 파일", self.new_file, "새로운 이미지를 만듭니다.", "./icons/new_file.png")
        self.add_action(file_menu, "열기", self.open_file, None, "./icons/open.png")
        self.add_action(file_menu, "저장", self.save_file, None,"./icons/save.png")
        file_menu.addSeparator()  # 구분선
        self.add_action(file_menu, "종료", self.close, None, "./icons/exit.png")

        # 편집 메뉴
        edit_menu = menu_bar.addMenu("편집")
        self.add_action(edit_menu, "되돌리기", self.undo, None, "./icons/undo.png")
        self.add_action(edit_menu, "다시 실행", self.redo, None, "./icons/redo.png")

        # 필터 메뉴
        filter_menu = menu_bar.addMenu("필터")
        self.add_action(filter_menu, "흑백 변환", self.convert_to_grayscale, "이미지를 흑백으로 변환합니다.", "./icons/grayscale.png")
        self.add_action(filter_menu, "어댑티브 쓰레시홀드", self.apply_adaptive_threshold, tooltip="이미지 밝기에 따라 어댑티브 쓰레시홀드 적용.")
        self.add_action(filter_menu, "스케치/페인팅 효과", self.apply_sketch_paint, tooltip="이미지를 스케치 및 페인팅 효과로 변환합니다.")
        self.add_action(filter_menu, "열림 연산", self.apply_opening, tooltip="이미지에 열림 연산(침식 후 팽창)을 적용합니다.")
        self.add_action(filter_menu, "닫힘 연산", self.apply_closing, tooltip="이미지에 닫힘 연산(팽창 후 침식)을 적용합니다.")
        self.add_action(filter_menu, "블러", self.apply_blur, tooltip="이미지에 블러 효과를 적용합니다.")
        self.add_action(filter_menu, "모폴로지 그레디언트", self.apply_morphological_gradient, tooltip="이미지에 모폴로지 그레디언트를 적용합니다.")
        self.add_action(filter_menu, "정규화(화질 개선)", self.apply_normalization, tooltip="이미지 밝기 및 대비를 정규화하여 화질을 개선합니다.")

        # 검출 메뉴
        detection_menu = menu_bar.addMenu("검출")
        self.add_action(detection_menu, "엣지 검출", self.apply_canny_edge_detection, tooltip="이미지에서 엣지를 검출합니다.")
        self.add_action(detection_menu, "코너 검출", self.apply_corner_detection, tooltip="이미지의 코너(모서리) 부분을 검출합니다.")
        self.add_action(detection_menu, "FAST 특징 검출", self.apply_fast_detection, tooltip="FAST 알고리즘을 사용하여 이미지 특징을 검출합니다.")
        self.add_action(detection_menu,"Bolb 검출", self.apply_blob_detection, tooltip="이미지에서 Blob 형태를 검출합니다.")

        # 도움말 메뉴
        help_menu = menu_bar.addMenu("도움말")
        self.add_action(help_menu, "정보", self.show_about, None, "./icons/info.png")

    def add_action(self, menu, name, handler, tooltip=None, icon_path=None):
        """메뉴에 액션을 추가."""
        action = QAction(name, self)
        if tooltip:
            # action.setStatusTip(tooltip)  # 상태 표시줄에 표시
            action.hovered.connect(lambda: QToolTip.showText(self.cursor().pos(), tooltip, menu))  # 툴팁 표시
        if icon_path:
            action.setIcon(QIcon(icon_path))
        action.triggered.connect(handler)
        menu.addAction(action)

    def show_about(self):
        """정보 창 표시."""
        QMessageBox.information(self, "정보", "소프트웨어학과_2020E7309_김선빈")

    def contextMenuEvent(self, event):
        """이미지 위에서 우클릭 시 팝업 메뉴를 표시"""
        if self.temp_image is None:
            return  # 이미지가 로드되지 않았다면 메뉴를 표시하지 않음

        menu = QMenu(self)

        # 팝업 메뉴 항목 추가
        actions = {
            "되돌리기": {"func": self.undo, "icon": "./icons/undo.png"},
            "다시 실행": {"func": self.redo, "icon": "./icons/redo.png"},
            "뒤집기(좌우)": {"func": self.flip_image_horizontal, "icon": "./icons/flip_image_horizontal.png"},
            "뒤집기(상하)": {"func": self.flip_image_vertical, "icon": "./icons/flip_image_vertical.png"},
            "이미지 저장": {"func": self.save_file, "icon": "./icons/save.png"},
        }

        for name, details in actions.items():
            if os.path.exists(details["icon"]):
                action = menu.addAction(QIcon(details["icon"]), name)
            else:
                action = menu.addAction(name)  # 아이콘 없이 추가
            action.triggered.connect(details["func"])

        menu.exec_(self.mapToGlobal(event.pos()))

    def new_file(self):
        """새 파일 생성."""
        # 작업 내역이 있을 경우 저장 여부 확인
        if self.history:
            response = QMessageBox.question(self, "작업 저장", "현재 작업을 저장하시겠습니까?",
                                            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if response == QMessageBox.Yes:
                self.save_file()
            elif response == QMessageBox.Cancel:
                return

        # 새 파일 대화창 열기
        dialog = NewFileDialog()
        if dialog.exec_() == QDialog.Accepted:
            width, height = dialog.get_dimensions()
            background_color = (255, 255, 255)  # 흰색 배경

            # 새 이미지 초기화
            self.temp_image = np.full((height, width, 3), background_color, dtype=np.uint8)
            self.image = self.temp_image.copy()
            self.scale_factor = 1.0  # 스케일 초기화

            # 히스토리 초기화
            self.history.clear()
            self.save_to_history()

            # 이미지 업데이트
            self.update_image()
            self.statusBar().showMessage(f"새 파일이 생성되었습니다. 크기: {width}x{height}")

    def crop_image(self):
        """이미지 자르기 도구"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # OpenCV ROI 선택
        cv2_temp_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)  # PyQt 이미지 -> OpenCV 이미지 변환
        roi = cv2.selectROI("Crop Image", cv2_temp_image, False)

        # ROI가 선택되었는지 확인
        if roi == (0, 0, 0, 0):
            print("유효한 영역이 선택되지 않았습니다.")
            cv2.destroyWindow("Crop Image")
            return

        x, y, w, h = roi
        cropped = cv2_temp_image[y:y + h, x:x + w]

        # 결과를 PyQt 이미지로 변환 (BGR -> RGB)
        self.update_and_save(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        cv2.destroyWindow("Crop Image")
        print("이미지 자르기 완료.")

    def create_panorama(self):
        """파노라마 이미지를 생성 ( 예제 8-37 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # 새로운 이미지를 선택
        file_path, _ = QFileDialog.getOpenFileName(self, "이미지 파일 열기", "", "Images (*.png *.jpg *.bmp)")
        if not file_path:
            print("파일을 선택하지 않았습니다.")
            return

        # 새 이미지 읽기
        img_array = np.fromfile(file_path, np.uint8)
        imgR = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if imgR is None:
            print("이미지를 읽을 수 없습니다.")
            return

        # OpenCV 처리: temp_image는 이미 로드된 이미지
        imgL = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)  # RGB -> BGR 변환
        hl, wl = imgL.shape[:2]
        hr, wr = imgR.shape[:2]

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # SIFT 특징 검출기 생성 및 특징점 검출
        descriptor = cv2.SIFT_create()
        kpsL, featuresL = descriptor.detectAndCompute(grayL, None)
        kpsR, featuresR = descriptor.detectAndCompute(grayR, None)

        # BF 매칭기 생성 및 knn매칭
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(featuresR, featuresL, 2)

        # 좋은 매칭점 선별
        good_matches = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                good_matches.append((m[0].trainIdx, m[0].queryIdx))

        # 좋은 매칭점이 4개 이상 원근 변환 행렬 구하기
        if len(good_matches) > 4:
            ptsL = np.float32([kpsL[i].pt for (i, _) in good_matches])
            ptsR = np.float32([kpsR[i].pt for (_, i) in good_matches])
            mtrx, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 4.0)

            # 확장된 크기의 결과 이미지 생성
            panorama_width = wl + wr
            panorama_height = max(hl, hr)
            panorama = cv2.warpPerspective(imgR, mtrx, (panorama_width, panorama_height))

            # 왼쪽 이미지를 합성
            panorama[0:hl, 0:wl] = imgL
        else:
            print("충분한 매칭 점이 없습니다. 기존 이미지를 반환합니다.")
            panorama = imgL

        # 파노라마 이미지를 BGR -> RGB 변환
        panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)

        # PyQt 이미지 업데이트
        self.update_and_save(panorama_rgb)
        print("파노라마 생성 완료.")

    def apply_orb_matching(self):
        """ORB 특징 매칭 ( 예제 8-19 )"""
        if self.temp_image is None:
            print("이미지를 로드하지 않았습니다.")
            return

        # PyQt 이미지 -> OpenCV 이미지로 변환 (RGB -> BGR)
        img1 = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # 새 이미지 선택
        file_path, _ = QFileDialog.getOpenFileName(self, "이미지 선택", "", "Images (*.png *.jpg *.bmp)")
        if not file_path:
            print("새로운 이미지를 선택하지 않았습니다.")
            return

        img_array = np.fromfile(file_path, np.uint8)
        img2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # img2 = cv2.imread(file_path)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # ORB로 서술자 추출
        detector = cv2.ORB_create()
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)

        # BFMatcher 생성 및 매칭
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)
        matches = matcher.knnMatch(desc1, desc2, k=2)

        # 좋은 매칭 선택
        ratio = 0.75
        good_matches = [first for first, second in matches if first.distance < second.distance * ratio]
        print(f"매칭된 점 개수: {len(good_matches)}/{len(matches)}")

        # 매칭 결과 이미지 그리기
        res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        # PyQt UI에 표시
        result_image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        self.update_and_save(result_image)
        print("ORB 매칭 결과가 표시되었습니다.")

    def apply_blob_detection(self):
        """Blob 검출 기능 ( 예제 8-9 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # OpenCV 이미지 (PyQt 이미지 -> BGR)
        bgr_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)

        # Grayscale 변환
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # Blob 검출 필터 파라미터 생성
        params = cv2.SimpleBlobDetector_Params()

        # 경계값 조정
        params.minThreshold = 10
        params.maxThreshold = 240
        params.thresholdStep = 5

        # 면적 필터 활성화 및 최소 값 지정
        params.filterByArea = True
        params.minArea = 200

        # 비활성화된 필터 설정
        params.filterByColor = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByCircularity = False

        # 필터 파라미터로 Blob 검출기 생성
        detector = cv2.SimpleBlobDetector_create(params)

        # 키 포인트 검출
        keypoints = detector.detect(gray_image)

        # 키 포인트를 이미지에 그리기
        detected_image = cv2.drawKeypoints(
            bgr_image, keypoints, None, (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # PyQt 이미지로 변환 및 화면 갱신
        self.update_and_save(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
        print(f"Blob 검출 완료. 검출된 Blob 개수: {len(keypoints)}")

    def apply_fast_detection(self):
        """FAST 특징 검출 기능 ( 예제 8-7 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # OpenCV 이미지 (PyQt 이미지 -> BGR)
        bgr_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)

        # 이미지를 Grayscale로 변환
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # FAST 특징 검출기 생성
        fast = cv2.FastFeatureDetector_create(threshold=50)

        # 키 포인트 검출
        keypoints = fast.detect(gray_image, None)

        # 키 포인트를 원본 이미지에 그리기
        detected_image = cv2.drawKeypoints(bgr_image, keypoints, None, color=(0, 255, 0))

        # PyQt 이미지로 변환 및 화면 갱신
        self.update_and_save(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
        print(f"FAST 특징 검출 완료. 검출된 특징 개수: {len(keypoints)}")

    def apply_corner_detection(self):
        """시-토마스 코너 검출 기능 ( 예제 8-5 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # OpenCV 이미지 (PyQt 이미지 -> BGR)
        bgr_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)

        # 이미지를 Grayscale로 변환
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # 시-토마스 코너 검출
        corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=80, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            corners = np.int32(corners)  # 좌표를 정수로 변환

            # 코너 위치에 동그라미 그리기
            for corner in corners:
                x, y = corner[0]
                cv2.circle(bgr_image, (x, y), 5, (0, 0, 255), 1, cv2.LINE_AA)

        # PyQt 이미지로 변환 및 화면 갱신
        self.update_and_save(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        print("시-토마스 코너 검출 완료.")

    def apply_template_matching(self):
        """템플릿 매칭 기능 ( 예제 8-3 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # 템플릿 이미지 선택
        file_path, _ = QFileDialog.getOpenFileName(self, "템플릿 이미지 선택", "", "Images (*.png *.xpm *.jpg *.bmp *.gif)")
        if not file_path:
            print("템플릿 이미지를 선택하지 않았습니다.")
            return

        # 템플릿 이미지 로드
        img_array = np.fromfile(file_path, np.uint8)
        template = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        #template = cv2.imread(file_path)
        if template is None:
            print("템플릿 이미지를 읽을 수 없습니다.")
            return

        # PyQt 이미지를 BGR 포맷으로 변환
        bgr_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)

        th, tw = template.shape[:2]  # 템플릿 크기
        methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED']

        results = []  # 매칭 결과 저장
        for method_name in methods:
            img_draw = bgr_image.copy()
            method = eval(method_name)
            res = cv2.matchTemplate(bgr_image, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
                match_val = min_val
            else:
                top_left = max_loc
                match_val = max_val

            bottom_right = (top_left[0] + tw, top_left[1] + th)
            cv2.rectangle(img_draw, top_left, bottom_right, (0, 0, 255), 2)
            cv2.putText(img_draw, f"{match_val:.2f}", top_left, cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, cv2.LINE_AA)

            results.append((method_name, match_val, img_draw))

        # 가장 높은 매칭 결과를 선택 (첫 번째 결과)
        _, _, best_result = results[0]

        # PyQt 이미지로 변환 및 화면 갱신
        self.update_and_save(cv2.cvtColor(best_result, cv2.COLOR_BGR2RGB))
        print("템플릿 매칭 완료.")

    def apply_document_scan(self):
        """문서 스캔 및 원근 변환 ( 예제 7-18 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # OpenCV 이미지를 BGR로 변환
        bgr_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)
        original = bgr_image.copy()

        # 그레이스케일 및 가우시안 블러
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 케니 엣지 검출
        edged = cv2.Canny(gray, 75, 200)

        # 컨투어 찾기
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in cnts:
            # 컨투어 근사
            peri = cv2.arcLength(c, True)
            vertices = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(vertices) == 4:
                break
        else:
            print("문서 형태를 감지하지 못했습니다.")
            return

        pts = vertices.reshape(4, 2)

        # 좌표 정렬
        sm = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        topLeft = pts[np.argmin(sm)]
        bottomRight = pts[np.argmax(sm)]
        topRight = pts[np.argmin(diff)]
        bottomLeft = pts[np.argmax(diff)]
        pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

        # 문서의 폭과 높이 계산
        w1 = np.linalg.norm(bottomRight - bottomLeft)
        w2 = np.linalg.norm(topRight - topLeft)
        h1 = np.linalg.norm(topRight - bottomRight)
        h2 = np.linalg.norm(topLeft - bottomLeft)
        width = int(max(w1, w2))
        height = int(max(h1, h2))

        # 변환 후 좌표
        pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

        # 원근 변환 행렬 계산 및 적용
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(original, mtrx, (width, height))

        # 변환된 이미지를 PyQt로 변환 후 업데이트
        self.update_and_save(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        print("문서 스캔 및 원근 변환 적용 완료.")

    def apply_grabcut(self):
        """GrabCut 알고리즘 ( 예제 7-15 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # OpenCV 이미지 준비
        bgr_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)
        img_draw = bgr_image.copy()
        rows, cols = bgr_image.shape[:2]
        mask = np.zeros((rows, cols), dtype=np.uint8)
        rect = [0, 0, 0, 0]
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)

        def on_mouse(event, x, y, flags, param):
            nonlocal rect, mask, img_draw

            if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 마우스 누름
                rect[:2] = x, y  # 시작 좌표 저장
                print(f"시작 좌표: {rect[:2]}")

            elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
                # 드래그 중 사각형 표시
                img_temp = img_draw.copy()
                cv2.rectangle(img_temp, (rect[0], rect[1]), (x, y), (0, 255, 0), 2)
                cv2.imshow("GrabCut", img_temp)

            elif event == cv2.EVENT_LBUTTONUP:  # 마우스 왼쪽 버튼 놓음
                rect[2:] = x, y  # 종료 좌표 저장
                print(f"종료 좌표: {rect[2:]}")
                cv2.rectangle(img_draw, (rect[0], rect[1]), (x, y), (255, 0, 0), 2)
                cv2.imshow("GrabCut", img_draw)

                # GrabCut 알고리즘 적용
                cv2.grabCut(bgr_image, mask, tuple(rect), bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_RECT)
                result = bgr_image.copy()
                # 확실한 배경과 아마도 배경을 0으로 설정
                result[(mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD)] = 0

                # 결과를 PyQt 이미지로 변환
                self.update_and_save(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                print("GrabCut 알고리즘 적용 완료.")
                cv2.destroyWindow("GrabCut")

        # GrabCut 창 및 마우스 이벤트 설정
        cv2.imshow("GrabCut", img_draw)
        cv2.setMouseCallback("GrabCut", on_mouse)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_watershed(self):
        """워터쉐드 알고리즘 ( 예제 7-14 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # OpenCV 이미지 준비
        bgr_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)
        rows, cols = bgr_image.shape[:2]
        img_draw = bgr_image.copy()

        # 마커 초기화
        marker = np.zeros((rows, cols), np.int32)
        marker_id = 1
        colors = []

        # 마우스 이벤트 처리 함수
        def on_mouse(event, x, y, flags, param):
            nonlocal img_draw, marker, marker_id, colors

            if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 누름
                # 마커 ID와 색상을 매핑
                colors.append((marker_id, tuple(map(int, bgr_image[y, x]))))
            elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:  # 드래그 중
                marker[y, x] = marker_id
                cv2.circle(img_draw, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow("Watershed", img_draw)
            elif event == cv2.EVENT_LBUTTONUP:  # 왼쪽 버튼 놓음
                marker_id += 1  # 다음 마커 ID로 이동
            elif event == cv2.EVENT_RBUTTONDOWN:  # 오른쪽 버튼 누름
                # 워터쉐드 알고리즘 실행
                cv2.watershed(bgr_image, marker)

                # 경계를 초록색으로 표시
                img_draw[marker == -1] = (0, 255, 0)
                for mid, color in colors:
                    img_draw[marker == mid] = color

                # 결과 표시
                cv2.imshow("Watershed", img_draw)

        # 창에서 마우스 이벤트 연결
        cv2.imshow("Watershed", img_draw)
        cv2.setMouseCallback("Watershed", on_mouse)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # PyQt 이미지로 변환
        self.update_and_save(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
        print("워터쉐드 알고리즘 적용 완료.")

    def apply_flood_fill(self):
        """Flood Fill 기능 ( 예제 7-13 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # PyQt 이미지 (RGB -> OpenCV BGR 이미지)
        bgr_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)

        # OpenCV 마우스 이벤트 처리 함수
        def onMouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # 현재 색상 가져오기
                new_val = self.brush_color[::-1]  # OpenCV는 BGR 순서이므로 순서 변경
                lo_diff, up_diff = (10, 10, 10), (10, 10, 10)  # 최소/최대 차이값

                # Flood fill 마스크 생성 (이미지보다 2픽셀 크게 생성)
                mask = np.zeros((bgr_image.shape[0] + 2, bgr_image.shape[1] + 2), np.uint8)

                # 색 채우기
                seed_point = (x, y)
                cv2.floodFill(bgr_image, mask, seed_point, new_val, lo_diff, up_diff)

                # 업데이트
                self.update_and_save(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
                print(f"Flood Fill 적용 완료: 색상 {new_val}")

        # OpenCV 창에서 마우스 이벤트 연결
        cv2.imshow("Flood Fill - 이미지 클릭", bgr_image)
        cv2.setMouseCallback("Flood Fill - 이미지 클릭", onMouse)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_sketch_paint(self):
        """스케치 및 페인팅 효과를 이미지에 적용 ( 예제 6-21 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # OpenCV 이미지 (RGB -> BGR)
        bgr_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)

        # 그레이 스케일로 변경
        img_gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        # 잡음 제거를 위해 가우시안 블러 필터 적용
        img_gray = cv2.GaussianBlur(img_gray, (9, 9), 0)
        # 라플라시안 필터로 엣지 검출
        edges = cv2.Laplacian(img_gray, -1, None, 5)
        # 스레시홀드로 경계 값 만 남기고 반전
        _, sketch = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)

        # 경계선 강조를 위해 팽창 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        sketch = cv2.erode(sketch, kernel)
        # 경계선 자연스럽게 하기 위해 미디언 블러 필터 적용
        sketch = cv2.medianBlur(sketch, 5)

        # 컬러 이미지에서 선명선을 없애기 위해 평균 블러 필터 적용
        img_paint = cv2.blur(bgr_image, (10, 10))
        # 컬러 영상과 스케치 영상과 합성
        img_result = cv2.bitwise_and(img_paint, img_paint, mask=sketch)

        # 결과를 PyQt 이미지로 변환 (BGR -> RGB)
        self.update_and_save(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
        print("스케치/페인팅 효과 적용 완료.")

    def apply_blur_mosaic(self):
        """블러(모자이크) 기능을 적용 ( 예제 6-20 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        self.current_mode = 'blur_mosaic'  # 모자이크 모드로 전환
        print("모자이크 모드를 활성화했습니다. 이미지를 드래그하여 영역을 선택하세요.")

        # # OpenCV 이미지 (RGB -> BGR)
        # bgr_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)
        #
        # # ROI 선택
        # (x, y, w, h) = cv2.selectROI("블러 모자이크 영역 선택", bgr_image, False)
        # cv2.destroyWindow("블러 모자이크 영역 선택")  # ROI 선택 창 닫기
        # if w == 0 or h == 0:
        #     print("유효한 ROI가 선택되지 않았습니다.")
        #     return
        #
        # # 관심 영역 지정
        # roi = bgr_image[y:y + h, x:x + w]
        #
        # # 블러(모자이크) 처리
        # ksize = 40  # 커널 크기 (필요에 따라 조정 가능)
        # blurred_roi = cv2.blur(roi, (ksize, ksize))
        #
        # # 블러 처리된 ROI를 원본 이미지에 적용
        # bgr_image[y:y + h, x:x + w] = blurred_roi
        #
        # # 결과를 PyQt 이미지로 변환 (BGR -> RGB)
        # self.update_and_save(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        # print("블러 모자이크 적용 완료.")

    def apply_blur_mosaic_effect(self):
        """모자이크 효과 적용 ( 예제 5-15 )"""
        if self.start_point and self.end_point and self.temp_image is not None:
            # 좌표 계산
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            x1, x2 = sorted((x1, x2))  # 정렬하여 좌상단, 우하단 순으로
            y1, y2 = sorted((y1, y2))

            # 선택 영역이 유효한지 확인
            if x1 == x2 or y1 == y2:
                print("유효하지 않은 선택 영역입니다.")
                return

            # OpenCV 이미지 좌표에 맞게 조정
            x1, y1 = int(x1 / self.scale_factor), int(y1 / self.scale_factor)
            x2, y2 = int(x2 / self.scale_factor), int(y2 / self.scale_factor)

            # OpenCV 이미지 (RGB -> BGR)
            bgr_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)

            # 관심 영역 지정
            roi = bgr_image[y1:y2, x1:x2]

            # 블러(모자이크) 처리
            ksize = 30  # 커널 크기
            blurred_roi = cv2.blur(roi, (ksize, ksize))

            # 블러 처리된 ROI를 원본 이미지에 적용
            bgr_image[y1:y2, x1:x2] = blurred_roi

            # 결과를 PyQt 이미지로 변환 (BGR -> RGB)
            self.temp_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            self.update_image()
            print(f"블러 모자이크 적용 완료: ({x1}, {y1})에서 ({x2}, {y2})")

    def apply_morphological_gradient(self):
        """모폴로지 그래디언트 연산 ( 예제 6-16 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # OpenCV 이미지는 BGR 형식이므로 채널 분리
        b, g, r = cv2.split(self.temp_image)

        # 구조화 요소 커널 생성 (3x3 사각형)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # 각 채널별로 모폴로지 그래디언트 연산 적용
        b_gradient = cv2.morphologyEx(b, cv2.MORPH_GRADIENT, kernel)
        g_gradient = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, kernel)
        r_gradient = cv2.morphologyEx(r, cv2.MORPH_GRADIENT, kernel)

        # 채널 병합
        gradient = cv2.merge((b_gradient, g_gradient, r_gradient))

        # 결과 저장 및 화면 갱신
        self.update_and_save(gradient)
        print("모폴로지 그래디언트 연산 적용 완료.")

    def apply_opening(self):
        """열림 연산을 적용 ( 예제 6-15 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

            # OpenCV 이미지는 BGR 형식이므로 채널 분리
        b, g, r = cv2.split(self.temp_image)

        # 구조화 요소 커널 생성 (5x5 사각형)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # 각 채널별로 열림 연산 적용
        b_opened = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)
        g_opened = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)
        r_opened = cv2.morphologyEx(r, cv2.MORPH_OPEN, kernel)

        # 채널 합치기
        opened = cv2.merge((b_opened, g_opened, r_opened))

        # 결과 저장 및 화면 갱신
        self.update_and_save(opened)
        print("열림 연산 적용 완료 (색상 유지).")

    def apply_closing(self):
        """닫힘 연산을 적용 ( 예제 6-15 ) """
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

            # OpenCV 이미지는 BGR 형식이므로 채널 분리
        b, g, r = cv2.split(self.temp_image)

        # 구조화 요소 커널 생성 (5x5 사각형)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # 각 채널별로 닫힘 연산 적용
        b_closed = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)
        g_closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel)
        r_closed = cv2.morphologyEx(r, cv2.MORPH_CLOSE, kernel)

        # 채널 합치기
        closed = cv2.merge((b_closed, g_closed, r_closed))

        # 결과 저장 및 화면 갱신
        self.update_and_save(closed)
        print("닫힘 연산 적용 완료 (색상 유지).")

    def apply_canny_edge_detection(self):
        """Canny Edge Detection ( 예제 6-12 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # PyQt 이미지 -> OpenCV 이미지 변환 (RGB -> Grayscale)
        gray_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2GRAY)

        # Canny Edge Detection 적용
        edges = cv2.Canny(gray_image, 100, 200)

        # 결과를 PyQt 이미지로 변환 (Grayscale -> RGB)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        self.update_and_save(edges_colored)
        print("Canny Edge Detection 적용 완료.")

    def apply_blur(self):
        """블러 효과를 적용 ( 예제 6-2 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # OpenCV 이미지 변환
        bgr_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)

        # 블러링 적용 (blur)
        blur = cv2.blur(bgr_image, (5, 5))

        # 결과를 PyQt 이미지로 변환
        self.update_and_save(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
        print("블러 효과 적용 완료.")

    def apply_liquify(self):
        """Liquify 효과를 적용 ( 예제 5-16 ) """
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # PyQt 이미지 (RGB) -> OpenCV 이미지 (BGR)
        bgr_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)
        liquify_image = bgr_image.copy()  # 원본 이미지 복사

        win_title = 'Liquify'
        half = 25
        isDragging = False
        cx1, cy1 = -1, -1

        def liquify(img, cx1, cy1, cx2, cy2):
            x, y, w, h = cx1 - half, cy1 - half, half * 2, half * 2

            # 이미지 경계를 벗어난 경우 처리
            if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
                print("선택 영역이 이미지 범위를 벗어났습니다.")
                return img

            roi = img[y:y + h, x:x + w].copy()
            out = roi.copy()

            offset_cx1, offset_cy1 = cx1 - x, cy1 - y
            offset_cx2, offset_cy2 = cx2 - x, cy2 - y

            tri1 = [
                [(0, 0), (w, 0), (offset_cx1, offset_cy1)],
                [(0, 0), (0, h), (offset_cx1, offset_cy1)],
                [(w, 0), (offset_cx1, offset_cy1), (w, h)],
                [(0, h), (offset_cx1, offset_cy1), (w, h)],
            ]

            tri2 = [
                [(0, 0), (w, 0), (offset_cx2, offset_cy2)],
                [(0, 0), (0, h), (offset_cx2, offset_cy2)],
                [(w, 0), (offset_cx2, offset_cy2), (w, h)],
                [(0, h), (offset_cx2, offset_cy2), (w, h)],
            ]

            for i in range(4):
                matrix = cv2.getAffineTransform(np.float32(tri1[i]), np.float32(tri2[i]))
                warped = cv2.warpAffine(roi.copy(), matrix, (w, h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT_101)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.int32(tri2[i]), (255, 255, 255))
                warped = cv2.bitwise_and(warped, warped, mask=mask)
                out = cv2.bitwise_and(out, out, mask=cv2.bitwise_not(mask))
                out = out + warped

            img[y:y + h, x:x + w] = out
            return img

        def onMouse(event, x, y, flags, param):
            nonlocal cx1, cy1, isDragging, liquify_image
            if event == cv2.EVENT_MOUSEMOVE:
                if not isDragging:
                    temp_img = liquify_image.copy()
                    cv2.rectangle(temp_img, (x - half, y - half), (x + half, y + half), (0, 255, 0))
                    cv2.imshow(win_title, temp_img)
            elif event == cv2.EVENT_LBUTTONDOWN:
                isDragging = True
                cx1, cy1 = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                if isDragging:
                    isDragging = False
                    liquify_image = liquify(liquify_image, cx1, cy1, x, y)
                    cv2.imshow(win_title, liquify_image)

        cv2.namedWindow(win_title)
        cv2.setMouseCallback(win_title, onMouse)
        cv2.imshow(win_title, liquify_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 결과를 PyQt 이미지로 변환
        self.update_and_save(cv2.cvtColor(liquify_image, cv2.COLOR_BGR2RGB))
        print("Liquify 효과 적용 완료.")

    def apply_default_mode(self):
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        self.current_mode = 'default'
        print("기본 모드로 변경했습니다.")

    def apply_mosaic(self):
        """이미지에 모자이크 효과 적용 ( 예제 5-15 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        self.current_mode = 'mosaic'  # 모자이크 모드로 전환
        print("모자이크 모드를 활성화했습니다. 이미지를 드래그하여 영역을 선택하세요.")

        # # PyQt 이미지 (RGB) -> OpenCV 이미지 (BGR)
        # bgr_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)
        #
        # # ROI 선택
        # (x, y, w, h) = cv2.selectROI("모자이크 영역 선택", bgr_image, False)
        # cv2.destroyWindow("모자이크 영역 선택")  # ROI 선택 창 닫기
        # if w == 0 or h == 0:
        #     print("유효한 ROI가 선택되지 않았습니다.")
        #     return
        #
        # # 모자이크 효과 적용
        # roi = bgr_image[y:y + h, x:x + w]
        # rate = 15  # 축소 비율
        # roi = cv2.resize(roi, (w // rate, h // rate))  # 축소
        # roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)  # 다시 확대
        # bgr_image[y:y + h, x:x + w] = roi  # 원본 이미지에 적용
        #
        # # 결과를 PyQt 이미지로 변환 (BGR -> RGB)
        # self.update_and_save(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        # print("모자이크 효과 적용 완료.")

    def apply_mosaic_effect(self):
        """선택된 영역에 모자이크 효과를 적용."""
        if self.start_point and self.end_point and self.temp_image is not None:
            # 좌표 계산
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            x1, x2 = sorted((x1, x2))  # 정렬하여 좌상단, 우하단 순으로
            y1, y2 = sorted((y1, y2))

            # 선택 영역이 유효한지 확인
            if x1 == x2 or y1 == y2:
                print("유효하지 않은 선택 영역입니다.")
                return

            # OpenCV 이미지 좌표에 맞게 조정
            x1, y1 = int(x1 / self.scale_factor), int(y1 / self.scale_factor)
            x2, y2 = int(x2 / self.scale_factor), int(y2 / self.scale_factor)

            # 모자이크 효과 적용
            roi = self.temp_image[y1:y2, x1:x2]
            rate = 15  # 축소 비율
            roi = cv2.resize(roi, (max(1, (x2 - x1) // rate), max(1, (y2 - y1) // rate)))  # 축소
            roi = cv2.resize(roi, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)  # 확대
            self.temp_image[y1:y2, x1:x2] = roi

            # 이미지 업데이트
            self.update_image()
            print(f"모자이크 적용 완료: ({x1}, {y1})에서 ({x2}, {y2})")

    def apply_radial_distortion(self):
        """이미지에 방사 왜곡 효과 적용 ( 예제 5-13 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # 방사 왜곡 계수 설정 (배럴 왜곡: 양수, 핀큐션 왜곡: 음수)
        # k1, k2, k3 = 0.5, 0.2, 0.0  # 배럴 왜곡
        k1, k2, k3 = -0.3, 0, 0    # 핀큐션 왜곡

        # 현재 이미지 크기 가져오기
        rows, cols = self.temp_image.shape[:2]

        # 매핑 배열 생성
        mapy, mapx = np.indices((rows, cols), dtype=np.float32)

        # 좌상단 기준 좌표를 -1~1 범위로 정규화 및 극좌표 변환
        mapx = 2 * mapx / (cols - 1) - 1
        mapy = 2 * mapy / (rows - 1) - 1
        r, theta = cv2.cartToPolar(mapx, mapy)

        # 방사 왜곡 변형 연산
        ru = r * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6))

        # 극좌표를 직교좌표로 변환 및 좌상단 기준으로 복원
        mapx, mapy = cv2.polarToCart(ru, theta)
        mapx = ((mapx + 1) * cols - 1) / 2
        mapy = ((mapy + 1) * rows - 1) / 2

        # 리매핑 변환
        distorted = cv2.remap(self.temp_image, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))

        # 결과 저장 및 화면 갱신
        self.update_and_save(distorted)
        print("방사 왜곡 효과 적용 완료.")

    def apply_distortion(self):
        """이미지에 볼록 렌즈 효과 적용 ( 예제 5-12 ) """
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # 설정 값
        exp = 2.0  # 볼록 렌즈 효과 지수 (1: 기본, >1: 볼록)
        scale = 1.0  # 왜곡 적용 범위 (0 ~ 1)

        # 현재 이미지 크기 가져오기
        rows, cols = self.temp_image.shape[:2]

        # 매핑 배열 생성
        mapy, mapx = np.indices((rows, cols), dtype=np.float32)

        # 좌상단 기준좌표를 -1~1 범위로 정규화
        mapx = 2 * mapx / (cols - 1) - 1
        mapy = 2 * mapy / (rows - 1) - 1

        # 중심에서 거리(r)와 각도(theta) 계산
        r, theta = cv2.cartToPolar(mapx, mapy)

        # 볼록 렌즈 효과를 중심에서 강하게 적용
        r = r ** exp

        # 극좌표를 직교좌표로 변환
        mapx, mapy = cv2.polarToCart(r, theta)

        # 좌표를 0 ~ cols 및 0 ~ rows 범위로 변환
        mapx = ((mapx + 1) * cols - 1) / 2
        mapy = ((mapy + 1) * rows - 1) / 2

        # 재매핑 변환
        distorted = cv2.remap(self.temp_image, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))

        # 결과 저장 및 화면 갱신
        self.update_and_save(distorted)
        print("볼록 렌즈 효과 적용 완료.")

    def flip_image_horizontal(self):
        """이미지를 좌우로 뒤집기 ( 예제 5-10 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        rows, cols = self.temp_image.shape[:2]
        mflip = np.float32([[-1, 0, cols - 1], [0, 1, 0]])  # 좌우 뒤집기 변환 행렬
        flipped = cv2.warpAffine(self.temp_image, mflip, (cols, rows))

        # 결과 저장 및 화면 업데이트
        self.update_and_save(flipped)
        print("이미지를 좌우로 뒤집었습니다.")

    def flip_image_vertical(self):
        """이미지를 상하로 뒤집기 ( 예제 5-10 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        rows, cols = self.temp_image.shape[:2]
        mflip = np.float32([[1, 0, 0], [0, -1, rows - 1]])  # 상하 뒤집기 변환 행렬
        flipped = cv2.warpAffine(self.temp_image, mflip, (cols, rows))

        # 결과 저장 및 화면 업데이트
        self.update_and_save(flipped)
        print("이미지를 상하로 뒤집었습니다.")

    def apply_back_projection(self):
        """ROI 기반으로 역투영 기능을 적용 ( 예제 4-32 ) """
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # PyQt 이미지 (RGB) -> OpenCV 이미지 (BGR)
        bgr_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)

        # ROI 선택
        (x, y, w, h) = cv2.selectROI("ROI 선택", bgr_image, False)
        cv2.destroyWindow("ROI 선택")  # 선택이 완료되면 창 닫기
        if w == 0 or h == 0:
            print("유효한 ROI가 선택되지 않았습니다.")
            return

        # ROI 설정 및 HSV 변환
        roi = bgr_image[y:y + h, x:x + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # ROI 히스토그램 계산
        hist_roi = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist_roi, hist_roi, 0, 255, cv2.NORM_MINMAX)

        # 역투영
        bp = cv2.calcBackProject([hsv_img], [0, 1], hist_roi, [0, 180, 0, 256], 1)

        # 마스킹 및 결과 계산
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.filter2D(bp, -1, disc, bp)
        _, mask = cv2.threshold(bp, 50, 255, cv2.THRESH_BINARY)
        result = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)

        # 결과를 PyQt 이미지로 변환 (BGR -> RGB)
        self.update_and_save(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        print("역투영 적용 완료.")

    def apply_normalization(self):
        """컬러 이미지를 유지하는 정규화 ( 예제 4-27 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # 각 채널별로 정규화 수행
        b, g, r = cv2.split(self.temp_image)  # BGR 채널 분리
        b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
        g_norm = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        r_norm = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)

        # 정규화된 채널 합치기
        img_norm = cv2.merge((b_norm, g_norm, r_norm))

        # 결과를 업데이트
        self.update_and_save(img_norm)
        print("컬러 이미지 정규화 완료.")

    def apply_chromakey(self):
        """크로마키 배경 제거 후 새로운 배경 이미지와 합성 ( 예제 4-23 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # 배경 이미지를 선택
        file_path, _ = QFileDialog.getOpenFileName(self, "배경 이미지 선택", "", "Images (*.png *.xpm *.jpg *.bmp *.gif)")
        if not file_path:
            print("배경 이미지를 선택하지 않았습니다.")
            return

        # 배경 이미지 읽기
        img_array = np.fromfile(file_path, np.uint8)
        bg_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # bg_image = cv2.imread(file_path)
        if bg_image is None:
            print("배경 이미지를 읽을 수 없습니다.")
            return

        # 현재 작업 중인 이미지
        fg_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR)
        fg_height, fg_width = fg_image.shape[:2]
        bg_height, bg_width = bg_image.shape[:2]

        # 배경 이미지를 크로마키 영상 크기로 조정
        resized_bg = cv2.resize(bg_image, (fg_width, fg_height), interpolation=cv2.INTER_AREA)

        # 크로마키 영역 정의 (상단 좌측 10x10 픽셀 기준)
        chromakey = fg_image[:10, :10, :]
        hsv_chroma = cv2.cvtColor(chromakey, cv2.COLOR_BGR2HSV)
        hsv_fg = cv2.cvtColor(fg_image, cv2.COLOR_BGR2HSV)

        # HSV 범위 설정
        chroma_h = hsv_chroma[:, :, 0]
        offset = 20

        lower = np.array([chroma_h.min() - offset, 100, 100])
        upper = np.array([chroma_h.max() + offset, 255, 255])

        # 마스크 생성
        mask = cv2.inRange(hsv_fg, lower, upper)
        mask_inv = cv2.bitwise_not(mask)

        # 전경과 배경 합성
        fg = cv2.bitwise_and(fg_image, fg_image, mask=mask_inv)
        bg = cv2.bitwise_and(resized_bg, resized_bg, mask=mask)
        combined = cv2.add(fg, bg)

        # 결과 저장 및 갱신
        self.update_and_save(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        print("크로마키 합성이 완료되었습니다.")

    def apply_alpha_blending(self):
        """알파 블렌딩을 적용하여 이미지를 합성 ( 예제 4-16 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # 파일 다이얼로그로 새로운 이미지 선택
        file_path, _ = QFileDialog.getOpenFileName(self, "이미지 파일 열기", "", "Images (*.png *.xpm *.jpg *.bmp *.gif)")
        if not file_path:
            print("이미지를 선택하지 않았습니다.")
            return

        # 새 이미지 읽기
        img_array = np.fromfile(file_path, np.uint8)
        new_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # new_image = cv2.imread(file_path)
        if new_image is None:
            print("이미지를 읽을 수 없습니다.")
            return

        # 현재 이미지 크기로 리사이즈
        height, width, _ = self.temp_image.shape
        new_image_resized = cv2.resize(new_image, (width, height), interpolation=cv2.INTER_AREA)

        # 알파 블렌딩 적용
        alpha = 0.5  # 알파 값 (필요 시 사용자 조정 가능)
        blended = cv2.addWeighted(self.temp_image, alpha, new_image_resized, 1 - alpha, 0)

        # 결과 저장 및 업데이트
        self.update_and_save(blended)
        print("알파 블렌딩 적용 완료.")

    def apply_adaptive_threshold(self):
        """이미지에 어댑티브 쓰레시홀드 ( 예제 4-12 )"""
        if self.temp_image is None:
            print("이미지가 로드되지 않았습니다.")
            return

        # OpenCV 이미지를 Grayscale로 변환
        gray = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2GRAY)

        blk_size = 9  # 블록 크기
        C = 5  # 차감 상수

        # 오츠의 알고리즘으로 단일 경계 값을 전체 이미지에 적용
        ret, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 어뎁티드 쓰레시홀드를 평균과 가우시안 분포로 각각 적용
        th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk_size, C)
        th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk_size, C)

        # Otsu's 알고리즘으로 변환된 이미지를 적용
        self.update_and_save(cv2.cvtColor(th1, cv2.COLOR_GRAY2RGB))
        print(f"Otsu's 알고리즘 적용 완료. Threshold 값: {ret}")

    def draw_text(self, image, position, text):
        """이미지에 텍스트를 그림."""
        font = cv2.FONT_HERSHEY_SIMPLEX  # 기본 글꼴
        font_scale = 1.0  # 글씨 크기
        thickness = 2  # 글씨 두께
        color = self.brush_color  # 현재 설정된 브러쉬 색상
        cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    def change_shape(self, index):
        """드롭다운에서 선택한 도형에 따라 current_shape 설정."""
        shapes = [None, 'line', 'circle', 'triangle', 'rectangle', 'text']  # 팬 도구와 도형 매핑
        self.current_shape = shapes[index]
        print(f"선택된 도형: {self.current_shape}")

        # "텍스트 그리기" 선택 시 입력창과 버튼 표시
        if self.current_shape == 'text':
            self.text_input.show()
            self.text_confirm_button.show()
        else:
            self.text_input.hide()
            self.text_confirm_button.hide()

    def confirm_text_input(self):
        """텍스트 입력 확인 버튼 클릭 시."""
        self.text_to_draw = self.text_input.text()  # 텍스트 저장
        print(f"입력한 텍스트: {self.text_to_draw}")
        self.text_input.hide()  # 입력창 숨김
        self.text_confirm_button.hide()  # 버튼 숨김

    def convert_to_grayscale(self):
        """이미지를 흑백으로 변환 ( 예제 4-6 )"""
        if self.temp_image is not None:
            self.temp_image = cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2GRAY)
            # 흑백 이미지를 컬러 포맷으로 변경 (QImage 업데이트를 위해 필요)
            self.update_and_save(cv2.cvtColor(self.temp_image, cv2.COLOR_GRAY2RGB))

    def add_sidebar_button(self, text, handler, tooltip=None):
        """사이드바에 버튼 추가."""
        button = QPushButton(text, self)
        button.clicked.connect(handler)
        button.setFixedWidth(200)
        if tooltip:
            button.setToolTip(tooltip)
        self.sidebar_layout.addWidget(button)

    def open_file(self):
        """이미지를 선택하고 로드."""
        if self.temp_image is not None and len(self.history) > 1:  # 이미지와 히스토리가 있는지 확인
            reply = QMessageBox.question(
                self, "저장 확인",
                "현재 이미지와 히스토리가 존재합니다. 저장하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if reply == QMessageBox.Yes:
                self.save_file()  # 현재 이미지를 저장
            elif reply == QMessageBox.Cancel:
                return  # 새 파일 열기를 취소

        # 새 파일 열기 진행
        file_path, _ = QFileDialog.getOpenFileName(self, "이미지 파일 열기", "", "Images (*.png *.xpm *.jpg *.bmp *.gif)")
        # np.fromfile 함수 사용, 바이너리 데이터를 넘파이 행렬로 읽음
        if file_path:
            img_array = np.fromfile(file_path, np.uint8)
            # cv2.imdecode 함수로 복호화
            self.image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.temp_image = self.image.copy()
            self.scale_factor = 1.0
            self.history.clear()  # 히스토리 초기화
            self.save_to_history()  # 새로운 상태를 히스토리에 저장
            self.update_image()

    def save_file(self):
        """이미지를 저장."""
        if self.image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "이미지 저장", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if file_path:
                # 확장자 확인
                extension = os.path.splitext(file_path)[1]
                # "OpenCV"에서 한글 경로 처리
                result, encoded_img = cv2.imencode(extension, cv2.cvtColor(self.temp_image, cv2.COLOR_RGB2BGR))
                if result:
                    # 한글 경로로 저장
                    with open(file_path, mode='w+b') as f:
                        encoded_img.tofile(f)

    def choose_color(self):
        """브러쉬 색상 선택."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.brush_color = (color.red(), color.green(), color.blue())
            # 색상 표시 업데이트
            self.color_display.setStyleSheet(f"background-color: rgb{self.brush_color}; border: 1px solid black;")

    def update_image(self):
        """QLabel에 이미지를 업데이트."""
        if self.temp_image is not None:
            h, w, c = self.temp_image.shape
            scaled_h, scaled_w = int(h * self.scale_factor), int(w * self.scale_factor)
            resized_image = cv2.resize(self.temp_image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            self.scaled_image = resized_image
            q_image = QImage(resized_image.data, resized_image.shape[1], resized_image.shape[0],
                             resized_image.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)

    def scale_image(self):
        """이미지 확대/축소."""
        if self.image is not None:
            self.scale_factor = self.scale_slider.value() / 100.0
            self.update_image()

    def save_to_history(self):
        """현재 상태를 작업 히스토리에 저장."""
        if self.temp_image is not None:
            self.history.append(self.temp_image.copy())
            self.redo_stack.clear() # 앞으로 가기 스택 초기화

    def undo(self):
        """작업 되돌리기."""
        if len(self.history) > 1:  # 현재 상태를 제외하고 이전 상태가 있을 경우
            self.redo_stack.append(self.history.pop()) # 현재 상태를 앞으로 가기 스택에 저장
            self.temp_image = self.history[-1].copy()  # 이전 상태 복원
            self.update_image()
        else:
            print("되돌릴 작업이 없습니다.")

    def redo(self):
        if self.redo_stack:  # 앞으로 가기 스택이 비어있지 않은 경우
            self.history.append(self.redo_stack.pop())  # 앞으로 가기 스택에서 상태 가져오기
            self.temp_image = self.history[-1].copy()  # 복원
            self.update_image()
        else:
            print("앞으로 갈 작업이 없습니다.")

    def mousePressEvent(self, event):
        """마우스 클릭 이벤트로 드로잉 시작."""
        if event.button() == Qt.LeftButton and self.image is not None:
            if self.current_mode == 'default':
                self.drawing = True
                self.start_point = self.map_to_image_coordinates(event.pos())  # 팬 도구나 도형의 시작점
                if self.current_shape is None:  # 팬 도구일 경우
                    self.last_point = self.start_point
                elif self.current_shape == 'text':  # 텍스트 그리기일 경우
                    if self.start_point and self.text_to_draw:  # 텍스트와 시작점이 유효한 경우
                        self.draw_text(self.temp_image, self.start_point, self.text_to_draw)
                        self.save_to_history()  # 작업 히스토리에 저장
                        self.update_image()  # 화면 갱신
                        print(f"텍스트 '{self.text_to_draw}'이(가) {self.start_point} 위치에 그려졌습니다.")
                    self.drawing = False  # 텍스트는 드래그가 없으므로 즉시 종료
            elif self.current_mode == 'mosaic':
                self.drawing = True
                self.start_point = self.map_to_image_coordinates(event.pos())  # 모자이크 시작 좌표
            elif self.current_mode == 'blur_mosaic':
                self.drawing = True
                self.start_point = self.map_to_image_coordinates(event.pos())  # 블러 모자이크 시작 좌표

    def mouseMoveEvent(self, event):
        """마우스 드래그로 팬 도구 또는 도형 그리기."""
        if event.buttons() == Qt.LeftButton and self.drawing:
            if self.current_mode == 'default':
                current_point = self.map_to_image_coordinates(event.pos())
                if current_point:
                    if self.current_shape is None:  # 팬 도구
                        cv2.line(self.temp_image, self.last_point, current_point, self.brush_color,
                                 self.brush_size_slider.value())
                        self.last_point = current_point
                        self.update_image()
                    else:  # 도형 그리기
                        self.end_point = current_point
                        temp_image = self.temp_image.copy()
                        self.draw_shape(temp_image, self.start_point, self.end_point, preview=True)
                        self.update_temp_image(temp_image)
            elif self.current_mode == 'mosaic':
                self.end_point = self.map_to_image_coordinates(event.pos())  # 모자이크 종료 좌표
                if self.start_point and self.end_point:
                    # 선택 영역을 시각적으로 표시 (예: 반투명 사각형)
                    temp_image = self.temp_image.copy()
                    x1, y1 = self.start_point
                    x2, y2 = self.end_point
                    cv2.rectangle(temp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 테두리
                    self.update_temp_image(temp_image)
            elif self.current_mode == 'blur_mosaic':
                self.end_point = self.map_to_image_coordinates(event.pos())  # 블러 모자이크 종료 좌표
                if self.start_point and self.end_point:
                    # 선택 영역 시각화 (예: 반투명 사각형)
                    temp_image = self.temp_image.copy()
                    x1, y1 = self.start_point
                    x2, y2 = self.end_point
                    cv2.rectangle(temp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 테두리
                    self.update_temp_image(temp_image)

    def mouseReleaseEvent(self, event):
        """마우스 버튼 해제 이벤트로 드로잉 종료."""
        if event.button() == Qt.LeftButton:
            if event.button() == Qt.LeftButton:
                if self.current_mode == 'default':
                    if self.drawing:
                        if self.current_shape is None:  # 팬 도구
                            self.save_to_history()  # 작업 히스토리에 저장
                            self.update_image()
                        elif self.start_point and self.end_point:  # 도형
                            self.draw_shape(self.temp_image, self.start_point, self.end_point)
                            self.save_to_history()  # 작업 히스토리에 저장
                            self.update_image()
                elif self.current_mode == 'mosaic':
                    if self.start_point and self.end_point:
                        # 모자이크 효과 적용
                        self.apply_mosaic_effect()
                        self.save_to_history()  # 작업 히스토리에 저장
                    self.current_mode = 'default'
                    self.drawing = False
                    self.start_point = None
                    self.end_point = None
                if self.current_mode == 'blur_mosaic':
                    if self.start_point and self.end_point:
                        # 블러 모자이크 효과 적용
                        self.apply_blur_mosaic_effect()
                        self.save_to_history()  # 작업 히스토리에 저장
                    self.current_mode = 'default'
                    self.drawing = False
                    self.start_point = None
                    self.end_point = None

    def map_to_image_coordinates(self, pos):
        """마우스 좌표를 이미지 좌표로 변환."""
        if self.scaled_image is None or self.image_label.pixmap() is None:
            return None

        # QLabel과 이미지의 크기
        label_rect = self.image_label.geometry()
        pixmap_rect = self.image_label.pixmap().rect()
        pixmap_rect.moveCenter(label_rect.center())

        # QMenuBar 높이 보정
        menu_bar_height = self.menuBar().height()

        # QLabel 내부의 이미지 시작 위치 계산
        x_offset = pixmap_rect.left()
        y_offset = pixmap_rect.top() + menu_bar_height  # 메뉴바 높이를 추가

        # 마우스 좌표를 이미지 좌표로 변환
        x = (pos.x() - x_offset) / self.scale_factor
        y = (pos.y() - y_offset) / self.scale_factor

        # 변환된 좌표가 이미지 범위 내에 있는지 확인
        if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
            return int(x), int(y)
        return None

    def draw_shape(self, image, start_point, end_point, preview=False):
        """도형을 그림."""
        color = self.brush_color if not preview else (245, 245, 245)  # 미리보기는 흰색
        thickness = 2 if preview else self.brush_size_slider.value()  # 두께 설정

        if self.current_shape == 'line':  # 선
            cv2.line(image, start_point, end_point, color, thickness)
        elif self.current_shape == 'circle':  # 원
            center = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
            radius = int(((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2) ** 0.5 / 2)
            cv2.circle(image, center, radius, color, thickness if preview else -1)
        elif self.current_shape == 'rectangle':  # 사각형
            cv2.rectangle(image, start_point, end_point, color, thickness if preview else -1)
        elif self.current_shape == 'triangle':  # 삼각형
            pts = np.array([
                [start_point[0], end_point[1]],
                [(start_point[0] + end_point[0]) // 2, start_point[1]],
                [end_point[0], end_point[1]]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            if not preview:
                cv2.fillPoly(image, [pts], color)
            else:
                cv2.polylines(image, [pts], True, color, thickness)

    def update_temp_image(self, temp_image):
        """QLabel에 임시 이미지를 업데이트."""
        h, w, c = temp_image.shape
        scaled_h, scaled_w = int(h * self.scale_factor), int(w * self.scale_factor)
        resized_image = cv2.resize(temp_image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        q_image = QImage(resized_image.data, resized_image.shape[1], resized_image.shape[0],
                         resized_image.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def update_and_save(self, processed_image):
        self.temp_image = processed_image
        self.save_to_history()
        self.update_image()

class NewFileDialog(QDialog):
    """새 파일 생성 대화창."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("새 파일 생성")
        self.setGeometry(200, 200, 350, 150)

        self.setWindowIcon(QIcon("./icons/photomate.png"))

        layout = QVBoxLayout()

        # 폭 입력
        width_layout = QHBoxLayout()
        width_label = QLabel("폭 (Width):")
        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText("800")  # 기본 값
        width_layout.addWidget(width_label)
        width_layout.addWidget(self.width_input)

        # 높이 입력
        height_layout = QHBoxLayout()
        height_label = QLabel("높이 (Height):")
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("600")  # 기본 값
        height_layout.addWidget(height_label)
        height_layout.addWidget(self.height_input)

        layout.addLayout(width_layout)
        layout.addLayout(height_layout)

        # 버튼
        button_layout = QHBoxLayout()
        confirm_button = QPushButton("확인")
        cancel_button = QPushButton("취소")
        confirm_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(confirm_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_dimensions(self):
        """사용자가 입력한 폭과 높이를 반환."""
        width = int(self.width_input.text()) if self.width_input.text() else 800
        height = int(self.height_input.text()) if self.height_input.text() else 600
        return width, height

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = PhotoEditor()
    editor.show()
    sys.exit(app.exec_())