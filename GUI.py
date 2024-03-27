import torch
import threading

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import QObject, pyqtSignal

from PyQt5 import QtCore  # core Qt functionality
from PyQt5 import QtGui  # extends QtCore with GUI functionality
from PyQt5 import QtOpenGL  # provides QGLWidget, a special OpenGL QWidget
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QPushButton, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QObject, pyqtSlot as Slot
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import pyqtSignal
import numpy as np
import sys

import pyqtgraph as pg

from lidarDataLoader import Lidar
from dataPreProcessor import preProcess, loadData
from trainers import Trainer

class FuncThread(threading.Thread):
    def __init__(self,t,*a):
        self._t=t
        self._a=a
        threading.Thread.__init__(self)

    def run(self):
        self._t(*self._a)

class ChartWindow(QMainWindow):
    updateScene = pyqtSignal(str)
    updateGLWindow = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('SDF Visualization')
        self.pointCoordinates = []
        self.obstaclePointCoordinates = []
        self.pointCloud = np.empty((0, 3), dtype=np.float32)
        self.ifBreak = False

        self._run_button = QPushButton("Run")
        self._break_button = QPushButton("Break")

        self.plot_widget_training_curve = pg.PlotWidget()
        self.plot_widget_object_visualizer = pg.PlotWidget()
        self.plot_widget_true_XY = pg.PlotWidget()
        self.plot_widget_true_YZ = pg.PlotWidget()
        self.plot_widget_true_XZ = pg.PlotWidget()
        self.plot_widget_predicted_XY = pg.PlotWidget()
        self.plot_widget_predicted_YZ = pg.PlotWidget()
        self.plot_widget_predicted_XZ = pg.PlotWidget()

        self.plot_widget_object_visualizer.setLabel('bottom', 'X')
        self.plot_widget_object_visualizer.setLabel('left', 'Y')
        self.plot_widget_object_visualizer.setLabel('top', 'Sign Distance Object Bird View')

        self.plot_widget_training_curve.setLabel('bottom', 'Epocs')
        self.plot_widget_training_curve.setLabel('left', 'Accuracy')
        self.plot_widget_training_curve.setLabel('top', 'Training curve')

        self.plot_widget_true_XY.setLabel('bottom', 'X')
        self.plot_widget_true_XY.setLabel('left', 'Y')
        self.plot_widget_true_XY.setLabel('top', 'True X-Y')

        self.plot_widget_true_YZ.setLabel('bottom', 'Y')
        self.plot_widget_true_YZ.setLabel('left', 'Z')
        self.plot_widget_true_YZ.setLabel('top', 'True Y-Z')

        self.plot_widget_true_XZ.setLabel('bottom', 'X')
        self.plot_widget_true_XZ.setLabel('left', 'Z')
        self.plot_widget_true_XZ.setLabel('top', 'True X-Z')

        self.plot_widget_predicted_XY.setLabel('bottom', 'X')
        self.plot_widget_predicted_XY.setLabel('left', 'Y')
        self.plot_widget_predicted_XY.setLabel('top', 'Predicted X-Y')

        self.plot_widget_predicted_YZ.setLabel('bottom', 'Y')
        self.plot_widget_predicted_YZ.setLabel('left', 'Z')
        self.plot_widget_predicted_YZ.setLabel('top', 'Predicted Y-Z')

        self.plot_widget_predicted_XZ.setLabel('bottom', 'X')
        self.plot_widget_predicted_XZ.setLabel('left', 'Z')
        self.plot_widget_predicted_XZ.setLabel('top', 'Predicted X-Z')

        self._run_button.clicked.connect(self.run)
        self._break_button.clicked.connect(self.breakAlgorithm)
        self.updateScene.connect(self.update)

        controller_widget = QWidget()
        controller_layout = QHBoxLayout(controller_widget)

        true_widget = QWidget()
        true_layout = QHBoxLayout(true_widget)

        predicted_widget = QWidget()
        predicted_layout = QHBoxLayout(predicted_widget)

        controller_layout.addWidget(self._run_button)
        controller_layout.addWidget(self._break_button)
        controller_layout.addWidget(self.plot_widget_training_curve)
        controller_layout.addWidget(self.plot_widget_object_visualizer)

        true_layout.addWidget(self.plot_widget_true_XY)
        true_layout.addWidget(self.plot_widget_true_YZ)
        true_layout.addWidget(self.plot_widget_true_XZ)

        predicted_layout.addWidget(self.plot_widget_predicted_XY)
        predicted_layout.addWidget(self.plot_widget_predicted_YZ)
        predicted_layout.addWidget(self.plot_widget_predicted_XZ)

        main_widget = QWidget(self)
        main_layout = QVBoxLayout(main_widget)
        main_layout.addWidget(controller_widget)
        main_layout.addWidget(true_widget)
        main_layout.addWidget(predicted_widget)
        main_layout.addStretch()
        self.setCentralWidget(main_widget)

        calculating_thread = FuncThread(self.update)
        calculating_thread.start()

    @Slot()
    def run(self):
        lidar = Lidar(self)
        lidar.load()

    @Slot()
    def breakAlgorithm(self):
        self.ifBreak = True

    def update(self):
        self.obstaclePointCoordinates, completed_dataset = preProcess(self.pointCoordinates)
        train_loader, test_loader = loadData(completed_dataset)
        model = Trainer(self, train_loader, test_loader)
        self.update_true_coordinates(self.obstaclePointCoordinates)
        self.update_predicted_coordinates(model, completed_dataset)
        QCoreApplication.processEvents()

    def update_true_coordinates(self, original_points):
        self.plot_widget_true_XY.clear()
        self.plot_widget_true_YZ.clear()
        self.plot_widget_true_XZ.clear()

        x_points = []
        y_points = []
        z_points = []
        for i in original_points:
            x_points.append(i[0])
            y_points.append(i[1])
            z_points.append(i[2])

        pen = pg.mkPen(color=(255, 10, 0))

        self.xy_data_line = pg.ScatterPlotItem(x_points, y_points, pen=pen, brush=pg.mkBrush(255, 0, 0, 128))
        self.yz_data_line = pg.ScatterPlotItem(y_points, z_points, pen=pen, brush=pg.mkBrush(255, 0, 0, 64))
        self.xz_data_line = pg.ScatterPlotItem(x_points, z_points, pen=pen, brush=pg.mkBrush(255, 0, 0, 64))
        self.xy_data_line.setSize(2)
        self.yz_data_line.setSize(2)
        self.xz_data_line.setSize(2)

        self.plot_widget_true_XY.addItem(self.xy_data_line)
        self.plot_widget_true_YZ.addItem(self.yz_data_line)
        self.plot_widget_true_XZ.addItem(self.xz_data_line)

    def update_predicted_coordinates(self, model, points):
        self.plot_widget_predicted_XY.clear()
        self.plot_widget_predicted_YZ.clear()
        self.plot_widget_predicted_XZ.clear()

        self.plot_widget_object_visualizer.clear()

        x_points = []
        y_points = []
        z_points = []
        x_SDF_vis_points = []
        y_SDF_vis_points = []
        for i in points:
            distance, confidence = self.sdf_function(model, i[0], i[1], i[2])
            if distance < 0.1 and distance >= 0:
                x_points.append(i[0])
                y_points.append(i[1])
                z_points.append(i[2])
            if confidence > 0.6 and distance * confidence <= 0:
                x_SDF_vis_points.append(i[0])
                y_SDF_vis_points.append(i[1])

        pen = pg.mkPen(color=(25, 0, 160))
        self.xy_pred_data_line = pg.ScatterPlotItem(x_points, y_points,  pen=pen, brush=pg.mkBrush(255, 0, 0, 128))
        self.yz_pred_data_line = pg.ScatterPlotItem(y_points, z_points, pen=pen, brush=pg.mkBrush(255, 0, 0, 64))
        self.xz_pred_data_line = pg.ScatterPlotItem(x_points, z_points, pen=pen, brush=pg.mkBrush(255, 0, 0, 64))
        self.xy_pred_data_line.setSize(2)
        self.yz_pred_data_line.setSize(2)
        self.xz_pred_data_line.setSize(2)

        self.plot_widget_predicted_XY.addItem(self.xy_pred_data_line)
        self.plot_widget_predicted_YZ.addItem(self.yz_pred_data_line)
        self.plot_widget_predicted_XZ.addItem(self.xz_pred_data_line)

        self.obj_line = self.plot_widget_object_visualizer.plot(x_SDF_vis_points, y_SDF_vis_points, pen=pen)

    def update_training_curve(self):
        print('')

    # def load_pointcloud(self, model, completed_dataset):
    #     self.plot_widget_object_visualizer.clear()
    #
    #     xPoints = []
    #     yPoints = []
    #     for i in completed_dataset:
    #         distance, confidence = self.sdf_function(model, i[0], i[1], i[2])
    #         if confidence > 0.6 and distance * confidence <= 0:
    #             pnt = np.array([i[0], i[1], i[2]], dtype=np.float32)
    #             xPoints.append(i[0])
    #             yPoints.append(i[1])
    #             self.pointCloud = np.vstack((self.pointCloud, pnt))
    #             #self.colors = np.vstack((self.colors, [0.7, 0.5, 0.4]))
    #
    #     pen = pg.mkPen(color=(255, 0, 0))
    #     self.data_line = self.plot_widget_object_visualizer.plot(xPoints, yPoints, pen=pen)
    #
    #
    #
    #
    #     print("size to gl: ",len(self.pointCloud))
    #     print("max to gl: ", np.max(self.pointCloud))
    #     print("min to gl: ", np.min(self.pointCloud))
    #     print("////////////////////////////////random points///////////////",self.pointCloud)

    def sdf_function(self, model,x, y, z):
        input_tensor = torch.tensor([x, y, z], dtype=torch.float32)
        with torch.no_grad():
            return model(input_tensor).detach().numpy()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = ChartWindow()
    win.resize(1400,800)
    #win.run()
    win.show()
    sys.exit(app.exec_())